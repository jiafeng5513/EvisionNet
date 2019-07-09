# -*- coding: utf-8 -*-
from __future__ import division
import time
import math
import tensorflow.contrib.slim as slim
import tensorflow as tf
import pprint
import random
import PIL.Image as pil
import scipy.misc
from pose_evaluation_utils import dump_pose_seq_TUM
from tensorflow.contrib.layers.python.layers import utils
from glob import glob
from data_loader import DataLoader
from data_loader import *
from utils import *
import re
from datetime import datetime

# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.01

TOWER_NAME = 'tower'

flags = tf.app.flags
flags.DEFINE_integer("run_mode", 2, "0=train,1=test_depth,2=test_pose")
flags.DEFINE_string("dataset_dir", "/home/RAID1/DataSet/KITTI/KittiOdometry/", "数据位置")
# KittiOdometry,KittiOdometry_prepared,KittiRaw,KittiRaw_prepared

# test pose  : KittiOdometry     /home/RAID1/DataSet/KITTI/KittiOdometry/
# test depth : KittiRaw
# train      : KittiRaw_prepared /home/RAID1/DataSet/KITTI/KittiRaw_prepared/

flags.DEFINE_string("checkpoint_dir", "../checkpoints/", "用于保存和加载ckpt的目录")

# params for model_train
flags.DEFINE_string("init_checkpoint_file", None, "用来初始化的ckpt")
flags.DEFINE_boolean("continue_train", False, "是否从之前的ckpt继续训练")

flags.DEFINE_float("learning_rate", 0.0002, "学习率")
flags.DEFINE_float("beta1", 0.9, "adam动量参数")
flags.DEFINE_float("smooth_weight", 0.5, "平滑的权重")
flags.DEFINE_float("explain_reg_weight", 0.2, "Weight for explanability regularization")
flags.DEFINE_integer("batch_size", 1, "batch size")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length", 3, "一个样本中含有几张图片")
flags.DEFINE_integer("num_source", 2, "一个样本中有几个是source images,这个值应该比seq_length少1")
flags.DEFINE_integer("max_steps", 200000, "训练迭代次数")
flags.DEFINE_integer("summary_freq", 100, "summary频率,单位:batch*num_gpus")
flags.DEFINE_integer("save_freq", 1000, "保存频率,单位:batch*num_gpus")
flags.DEFINE_integer('num_gpus', 3, "使用多少GPU")
flags.DEFINE_integer('num_epochs', 30, "把整个训练集训练多少次")
# params for model_test_depth
flags.DEFINE_string("test_file_list", '../data/kitti/test_files_eigen.txt', "Path to the list of test files")
flags.DEFINE_float("min_depth", 1e-3, "Threshold for minimum depth")
flags.DEFINE_float("max_depth", 80, "Threshold for maximum depth")

# params for model_test_pose
flags.DEFINE_integer("test_seq", 9, "使用KittiOdometry的哪个序列进行测试")  # pick from 22 sequences in KittiOdometry
flags.DEFINE_string("output_dir", "../test_output/test_pose/", "Output directory")

# add by jiafeng5513,followed by https://github.com/tinghuiz/SfMLearner/pull/70
flags.DEFINE_integer("num_scales", 4, "number of used image scales")

FLAGS = flags.FLAGS

GPU_ID = [1,2,3]  # 0,1,2,3

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.

    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_with_weight_decay(name, shape, stddev, wd):
    """创建一个带有权重衰减的初始化变量。注意变量是用截断的正态分布初始化的。 仅在指定了权重衰减时才添加权重衰减。
    Args:
      name:   变量名
      shape:  变量shape, list of ints
      stddev: 截断高斯分布的标准差
      wd:     添加L2Loss权重衰减,乘以此浮点数.如果为None,则不为此变量添加权重衰减.
    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_cpu(name, shape, initializer):
    """创建在主存(CPU内存)中存储的变量.
    Args:
      name: 变量名
      shape: list of ints
      initializer: 初始化器
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _conv2d_with_relu(input_data, out_channels, filter_height, filter_width, stride, name, use_activation=True):
    with tf.variable_scope(name) as scope:
        in_channels = input_data.get_shape().as_list()[3]  # input[batch_size, in_height, in_width, in_channels]
        kernel = _variable_with_weight_decay('weights', shape=[filter_height, filter_width, in_channels, out_channels],
                                             stddev=5e-2, wd=0.0)
        biases = _variable_on_cpu('biases', [out_channels], tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_data, kernel, strides=[1, stride, stride, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        if use_activation:
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv1)
            return conv1
        return pre_activation


def _conv2d_with_sigmoid(input_data, out_channels, filter_height, filter_width, stride, name, use_activation=True):
    with tf.variable_scope(name) as scope:
        in_channels = input_data.get_shape().as_list()[3]  # input[batch_size, in_height, in_width, in_channels]
        kernel = _variable_with_weight_decay('weights', shape=[filter_height, filter_width, in_channels, out_channels],
                                             stddev=5e-2, wd=0.0)
        biases = _variable_on_cpu('biases', [out_channels], tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_data, kernel, strides=[1, stride, stride, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        if use_activation:
            conv1 = tf.nn.sigmoid(pre_activation, name=scope.name)
            _activation_summary(conv1)
            return conv1
        return pre_activation


def _conv2d_transpose(input_data, out_channels, filter_height, filter_width, stride, name, use_activation=True):
    with tf.variable_scope(name) as scope:
        in_channels = input_data.get_shape().as_list()[3]  # input[batch_size, in_height, in_width, in_channels]
        kernel = _variable_with_weight_decay('weights', shape=[filter_height, filter_width, out_channels, in_channels],
                                             stddev=5e-2, wd=0.0)
        biases = _variable_on_cpu('biases', [out_channels], tf.constant_initializer(0.0))

        transpose_conv = tf.nn.conv2d_transpose(value=input_data, filter=kernel,
                                                output_shape=[input_data.get_shape().as_list()[0],
                                                              input_data.get_shape().as_list()[1] * 2,
                                                              input_data.get_shape().as_list()[2] * 2, out_channels],
                                                strides=[1, stride, stride, 1], padding='SAME')

        pre_activation = tf.nn.bias_add(transpose_conv, biases)

        if use_activation:
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv1)
            return conv1
        return pre_activation


def pose_exp_net(tgt_image, src_image_stack, do_exp=True, is_training=True):
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    num_source = int(src_image_stack.get_shape()[3].value // 3)
    with tf.variable_scope('pose_exp_net') as sc:
        cnv1 = _conv2d_with_relu(inputs, 16, 7, 7, 2, 'conv1')
        cnv2 = _conv2d_with_relu(cnv1, 32, 5, 5, 2, 'cnv2')
        cnv3 = _conv2d_with_relu(cnv2, 64, 3, 3, 2, 'cnv3')
        cnv4 = _conv2d_with_relu(cnv3, 128, 3, 3, 2, 'cnv4')
        cnv5 = _conv2d_with_relu(cnv4, 256, 3, 3, 2, 'cnv5')

        with tf.variable_scope('pose'):
            cnv6 = _conv2d_with_relu(cnv5, 256, 3, 3, 2, 'cnv6')
            cnv7 = _conv2d_with_relu(cnv6, 256, 3, 3, 2, 'cnv7')
            pose_pred = _conv2d_with_relu(cnv7, 6 * num_source, 1, 1, 1, 'pred', use_activation=False)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            # Empirically we found that scaling by a small constant
            # facilitates training.
            pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
            # Exp mask specific layers
            if do_exp:
                with tf.variable_scope('exp'):
                    upcnv5 = _conv2d_transpose(cnv5, 256, 3, 3, 2, 'upcnv5')
                    upcnv4 = _conv2d_transpose(upcnv5, 128, 3, 3, 2, 'upcnv4')
                    mask4 = _conv2d_with_relu(upcnv4, num_source * 2, 3, 3, 1, 'mask4', use_activation=False)
                    upcnv3 = _conv2d_transpose(upcnv4, 64, 3, 3, 2, 'upcnv3')
                    mask3 = _conv2d_with_relu(upcnv3, num_source * 2, 3, 3, 1, 'mask3', use_activation=False)
                    upcnv2 = _conv2d_transpose(upcnv3, 32, 5, 5, 2, 'upcnv2')
                    mask2 = _conv2d_with_relu(upcnv2, num_source * 2, 5, 5, 1, 'mask2', use_activation=False)
                    upcnv1 = _conv2d_transpose(upcnv2, 16, 7, 7, 2, 'upcnv1')
                    mask1 = _conv2d_with_relu(upcnv1, num_source * 2, 7, 7, 1, 'mask1', use_activation=False)
            else:
                mask1 = None
                mask2 = None
                mask3 = None
                mask4 = None
            return pose_final, [mask1, mask2, mask3, mask4]


def disp_net(tgt_image, is_training=True):
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net') as sc:
        cnv1 = _conv2d_with_relu(tgt_image, 32, 7, 7, 2, 'cnv1')
        cnv1b = _conv2d_with_relu(cnv1, 32, 7, 7, 1, 'cnv1b')
        cnv2 = _conv2d_with_relu(cnv1b, 64, 5, 5, 2, 'cnv2')
        cnv2b = _conv2d_with_relu(cnv2, 64, 5, 5, 1, 'cnv2b')
        cnv3 = _conv2d_with_relu(cnv2b, 128, 3, 3, 2, 'cnv3')
        cnv3b = _conv2d_with_relu(cnv3, 128, 3, 3, 1, 'cnv3b')
        cnv4 = _conv2d_with_relu(cnv3b, 256, 3, 3, 2, 'cnv4')
        cnv4b = _conv2d_with_relu(cnv4, 256, 3, 3, 1, 'cnv4b')
        cnv5 = _conv2d_with_relu(cnv4b, 512, 3, 3, 2, 'cnv5')
        cnv5b = _conv2d_with_relu(cnv5, 512, 3, 3, 1, 'cnv5b')
        cnv6 = _conv2d_with_relu(cnv5b, 512, 3, 3, 2, 'cnv6')
        cnv6b = _conv2d_with_relu(cnv6, 512, 3, 3, 1, 'cnv6b')
        cnv7 = _conv2d_with_relu(cnv6b, 512, 3, 3, 2, 'cnv7')
        cnv7b = _conv2d_with_relu(cnv7, 512, 3, 3, 1, 'cnv7b')

        upcnv7 = _conv2d_transpose(cnv7b, 512, 3, 3, 2, 'upcnv7')
        # There might be dimension mismatch due to uneven down/up-sampling
        upcnv7 = resize_like(upcnv7, cnv6b)
        i7_in = tf.concat([upcnv7, cnv6b], axis=3)
        icnv7 = _conv2d_with_relu(i7_in, 512, 3, 3, 1, 'icnv7')

        upcnv6 = _conv2d_transpose(icnv7, 512, 3, 3, 2, 'upcnv6')
        upcnv6 = resize_like(upcnv6, cnv5b)
        i6_in = tf.concat([upcnv6, cnv5b], axis=3)
        icnv6 = _conv2d_with_relu(i6_in, 512, 3, 3, 1, 'icnv6')

        upcnv5 = _conv2d_transpose(icnv6, 256, 3, 3, 2, 'upcnv5')
        upcnv5 = resize_like(upcnv5, cnv4b)
        i5_in = tf.concat([upcnv5, cnv4b], axis=3)
        icnv5 = _conv2d_with_relu(i5_in, 256, 3, 3, 1, 'icnv5')

        upcnv4 = _conv2d_transpose(icnv5, 128, 3, 3, 2, 'upcnv4')
        i4_in = tf.concat([upcnv4, cnv3b], axis=3)
        icnv4 = _conv2d_with_relu(i4_in, 128, 3, 3, 1, 'icnv4')
        disp4 = DISP_SCALING * _conv2d_with_sigmoid(icnv4, 1, 3, 3, 1, 'disp4') + MIN_DISP
        disp4_up = tf.image.resize_bilinear(disp4, [np.int(H / 4), np.int(W / 4)])

        upcnv3 = _conv2d_transpose(icnv4, 64, 3, 3, 2, 'upcnv3')
        i3_in = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
        icnv3 = _conv2d_with_relu(i3_in, 64, 3, 3, 1, 'icnv3')
        disp3 = DISP_SCALING * _conv2d_with_sigmoid(icnv3, 1, 3, 3, 1, 'disp3') + MIN_DISP
        disp3_up = tf.image.resize_bilinear(disp3, [np.int(H / 2), np.int(W / 2)])

        upcnv2 = _conv2d_transpose(icnv3, 32, 3, 3, 2, 'upcnv2')
        i2_in = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
        icnv2 = _conv2d_with_relu(i2_in, 32, 3, 3, 1, 'icnv2')
        disp2 = DISP_SCALING * _conv2d_with_sigmoid(icnv2, 1, 3, 3, 1, 'disp2') + MIN_DISP
        disp2_up = tf.image.resize_bilinear(disp2, [H, W])

        upcnv1 = _conv2d_transpose(icnv2, 16, 3, 3, 2, 'upcnv1')
        i1_in = tf.concat([upcnv1, disp2_up], axis=3)
        icnv1 = _conv2d_with_relu(i1_in, 16, 3, 3, 1, 'icnv1')
        disp1 = DISP_SCALING * _conv2d_with_sigmoid(icnv1, 1, 3, 3, 1, 'disp1') + MIN_DISP

        return [disp1, disp2, disp3, disp4]


def get_reference_explain_mask(downscaling):
    tmp = np.array([0, 1])
    ref_exp_mask = np.tile(tmp, (FLAGS.batch_size, int(FLAGS.img_height / (2 ** downscaling)),
                                 int(FLAGS.img_width / (2 ** downscaling)), 1))
    ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
    return ref_exp_mask


def compute_exp_reg_loss(pred, ref):
    l = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(ref, [-1, 2]),
        logits=tf.reshape(pred, [-1, 2]))
    return tf.reduce_mean(l)


def compute_smooth_loss(pred_disp):
    def gradient(pred):
        D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        return D_dx, D_dy

    dx, dy = gradient(pred_disp)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    return tf.reduce_mean(tf.abs(dx2)) + \
           tf.reduce_mean(tf.abs(dxdy)) + \
           tf.reduce_mean(tf.abs(dydx)) + \
           tf.reduce_mean(tf.abs(dy2))


def loss(tgt_image, src_image_stack, intrinsics, pred_disp, pred_poses, pred_exp_logits):
    """

    :param tgt_image:
    :param src_image_stack:
    :param intrinsics:
    :param pred_disp:
    :param pred_poses:
    :param pred_exp_logits:
    :return:
    """
    with tf.name_scope("compute_loss"):
        pixel_loss = 0
        exp_loss = 0
        smooth_loss = 0
        tgt_image_all = []
        src_image_stack_all = []
        proj_image_stack_all = []
        proj_error_stack_all = []
        exp_mask_stack_all = []
        pred_depth = [1. / d for d in pred_disp]
        for s in range(FLAGS.num_scales):
            if FLAGS.explain_reg_weight > 0:
                # Construct a reference explainability mask (i.e. all
                # pixels are explainable)
                ref_exp_mask = get_reference_explain_mask(s)
            # Scale the source and target images for computing loss at the
            # according scale.
            curr_tgt_image = tf.image.resize_area(tgt_image,
                                                  [int(FLAGS.img_height / (2 ** s)), int(FLAGS.img_width / (2 ** s))])
            curr_src_image_stack = tf.image.resize_area(src_image_stack,
                                                        [int(FLAGS.img_height / (2 ** s)),
                                                         int(FLAGS.img_width / (2 ** s))])

            if FLAGS.smooth_weight > 0:
                smooth_loss += FLAGS.smooth_weight / (2 ** s) * \
                               compute_smooth_loss(pred_disp[s])

            for i in range(FLAGS.num_source):
                # Inverse warp the source image to the target image frame
                curr_proj_image = projective_inverse_warp(
                    curr_src_image_stack[:, :, :, 3 * i:3 * (i + 1)],
                    tf.squeeze(pred_depth[s], axis=3),
                    pred_poses[:, i, :],
                    intrinsics[:, s, :, :])
                curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)
                # Cross-entropy loss as regularization for the
                # explainability prediction
                if FLAGS.explain_reg_weight > 0:
                    curr_exp_logits = tf.slice(pred_exp_logits[s],
                                               [0, 0, 0, i * 2],
                                               [-1, -1, -1, 2])
                    exp_loss += FLAGS.explain_reg_weight * compute_exp_reg_loss(curr_exp_logits, ref_exp_mask)
                    curr_exp = tf.nn.softmax(curr_exp_logits)
                # Photo-consistency loss weighted by explainability
                if FLAGS.explain_reg_weight > 0:
                    pixel_loss += tf.reduce_mean(curr_proj_error * tf.expand_dims(curr_exp[:, :, :, 1], -1))
                else:
                    pixel_loss += tf.reduce_mean(curr_proj_error)
                    # Prepare images for tensorboard summaries
                if i == 0:
                    proj_image_stack = curr_proj_image
                    proj_error_stack = curr_proj_error
                    if FLAGS.explain_reg_weight > 0:
                        exp_mask_stack = tf.expand_dims(curr_exp[:, :, :, 1], -1)
                else:
                    proj_image_stack = tf.concat([proj_image_stack, curr_proj_image], axis=3)
                    proj_error_stack = tf.concat([proj_error_stack, curr_proj_error], axis=3)
                    if FLAGS.explain_reg_weight > 0:
                        exp_mask_stack = tf.concat([exp_mask_stack, tf.expand_dims(curr_exp[:, :, :, 1], -1)], axis=3)
            tgt_image_all.append(curr_tgt_image)
            src_image_stack_all.append(curr_src_image_stack)
            proj_image_stack_all.append(proj_image_stack)
            proj_error_stack_all.append(proj_error_stack)
            if FLAGS.explain_reg_weight > 0:
                exp_mask_stack_all.append(exp_mask_stack)
        total_loss = pixel_loss + smooth_loss + exp_loss

        tf.add_to_collection('losses', total_loss)

        return total_loss


def tower_loss(tgt_image, src_image_stack, intrinsics, scope):
    """计算单卡上的损失值
    Args:
      scope: 指定显卡的唯一标识前缀, e.g. 'tower_0'
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """


    with tf.name_scope("depth_prediction"):
        pred_disp = disp_net(tgt_image, is_training=True)

    with tf.name_scope("pose_and_explainability_prediction"):
        pred_poses, pred_exp_logits = pose_exp_net(tgt_image, src_image_stack,
                                                   do_exp=(FLAGS.explain_reg_weight > 0), is_training=True)

    # 损失计算

    total_loss = loss(tgt_image, src_image_stack, intrinsics, pred_disp, pred_poses, pred_exp_logits)

    # Attach a scalar summary to all individual losses and the total loss; do
    # the same for the averaged version of the losses.

    tf.summary.scalar(scope + '_total_loss', total_loss, family='total_loss')

    return total_loss


def average_gradients(tower_grads):
    """计算在各个GPU中共享的变量的平均梯度,这是程序的同步点
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been
           averaged across all towers.
        """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # 训练次数的计数变量,train()每被调用一次就+1,global_step = batches processed * FLAGS.num_gpus
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # 学习率
        # num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)
        # decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        # # Decay the learning rate exponentially based on the number of steps.
        # lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
        #                                 global_step,
        #                                 decay_steps,
        #                                 LEARNING_RATE_DECAY_FACTOR,
        #                                 staircase=True)
        # 使用动量优化器,目前是固定学习率,TODO:策略学习率
        # opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True, use_locking=True)
        opt = tf.train.AdamOptimizer(FLAGS.learning_rate, FLAGS.beta1, use_locking=True)

        loader = DataLoader(FLAGS.dataset_dir, FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width,
                            FLAGS.num_source, FLAGS.num_scales)
        num_of_batch_in_an_epoch, num_train_examples, num_val_examples = loader.data_statistics()
        total_step = (FLAGS.num_epochs * num_of_batch_in_an_epoch) // FLAGS.num_gpus

        num_step_in_an_epoch = num_of_batch_in_an_epoch // FLAGS.num_gpus


        tgt_image, src_image_stack, intrinsics = loader.load_train_batch()
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [tgt_image, src_image_stack, intrinsics], capacity=2 * FLAGS.num_gpus)
        # 每个GPU分别计算梯度
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            #for i in range(FLAGS.num_gpus):
            for i in GPU_ID:
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:

                        tgt_image, src_image_stack, intrinsics = batch_queue.dequeue()
                        # 计算损失,注意,数据的加载,预测值,损失计算都包含在其中
                        _loss = tower_loss(tgt_image, src_image_stack, intrinsics, scope)
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
                        # 计算此批数据的梯度
                        grads = opt.compute_gradients(_loss, gate_gradients=0)
                        # 跟踪所有卡的梯度。
                        tower_grads.append(grads)

        # 所有卡梯度求均值(多卡同步点)
        grads = average_gradients(tower_grads)

        # 梯度变化图

        for grad, var in grads:
            if grad is not None:
                summaries.append(
                    tf.summary.histogram(var.op.name + '/gradients', grad))
        # Add a summary to track the learning rate.
        summaries.append(
            tf.summary.scalar('learning_rate', FLAGS.learning_rate, family='global'))  # 目前使用的是固定学习率 TODO:策略学习率

        # 应用梯度
        train_op = opt.apply_gradients(grads, global_step=global_step)


        # 可训练变量的曲线图

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), sharded=True)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        # init = tf.global_variables_initializer()

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Start running operations on the Graph. allow_soft_placement must be
        # set to True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir, sess.graph)
        print("Training in process...")
        try:
            step = 0
            # saver = tf.train.Saver([var for var in tf.model_variables()] + [global_step],max_to_keep=10)
            while not coord.should_stop():
                start_time = time.time()

                # 一步训练=4 batch
                _, loss_value = sess.run([train_op, _loss])
                # 计算用时
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                # TensorBoard
                if step % 30 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                # 终端输出
                current_epoch = step * FLAGS.num_gpus // num_of_batch_in_an_epoch + 1
                if step % (num_step_in_an_epoch // 10) == 0:
                    num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / FLAGS.num_gpus
                    format_str = '%s: epoch = %3d, step=(%d/%d,%.3f%%), loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                    print(format_str % (datetime.now(), current_epoch, step,
                                        total_step,(step/total_step)*100, loss_value, examples_per_sec, sec_per_batch))
                # 模型保存
                if (step != 0) and (step % num_step_in_an_epoch == 0):  # an epoch is done
                    print(" [*] %3dth epoch done! Saving latest checkpoint to %s..." % (current_epoch, FLAGS.checkpoint_dir))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model'), global_step=step)

                if step * FLAGS.num_gpus >= FLAGS.num_epochs * num_of_batch_in_an_epoch:
                    print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model.latest'))
                    coord.request_stop()
                step += 1
                #
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
    pass


def build_pose_test_graph():
    """
    build pose net for test
    :return: input placeholder and predicted poses
    """
    input_uint8 = tf.placeholder(tf.uint8, [FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width * FLAGS.seq_length, 3],
                                 name='raw_input')
    input_mc = preprocess_image(input_uint8)
    tgt_image, src_image_stack = batch_unpack_image_sequence(input_mc, FLAGS.img_height, FLAGS.img_width, FLAGS.num_source)
    with tf.name_scope("pose_prediction"):
        pred_poses, _ = pose_exp_net(tgt_image, src_image_stack, do_exp=False, is_training=False)
    return input_uint8, pred_poses


def build_depth_test_graph():
    """
    build depth net for test
    :return:
    """
    input_uint8 = tf.placeholder(tf.uint8, [FLAGS.batch_size,FLAGS.img_height, FLAGS.img_width, 3], name='raw_input')
    input_mc = preprocess_image(input_uint8)
    with tf.name_scope("depth_prediction"):
        pred_disp = disp_net(input_mc, is_training=False)
        pred_depth = [1. / disp for disp in pred_disp]
    pred_depth = pred_depth[0]
    return input_uint8, pred_depth


class EvisionNet(object):
    def __init__(self):
        pass


    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_loss", self.pixel_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        tf.summary.scalar("exp_loss", self.exp_loss)
        for s in range(opt.num_scales):
            tf.summary.histogram("scale%d_depth" % s, self.pred_depth[s])
            tf.summary.image('scale%d_disparity_image' % s, 1. / self.pred_depth[s])
            tf.summary.image('scale%d_target_image' % s, \
                             self.deprocess_image(self.tgt_image_all[s]))
            for i in range(opt.num_source):
                if opt.explain_reg_weight > 0:
                    tf.summary.image(
                        'scale%d_exp_mask_%d' % (s, i),
                        tf.expand_dims(self.exp_mask_stack_all[s][:, :, :, i], -1))
                tf.summary.image(
                    'scale%d_source_image_%d' % (s, i),
                    self.deprocess_image(self.src_image_stack_all[s][:, :, :, i * 3:(i + 1) * 3]))
                tf.summary.image('scale%d_projected_image_%d' % (s, i),
                                 self.deprocess_image(self.proj_image_stack_all[s][:, :, :, i * 3:(i + 1) * 3]))
                tf.summary.image('scale%d_proj_error_%d' % (s, i),
                                 self.deprocess_image(
                                     tf.clip_by_value(self.proj_error_stack_all[s][:, :, :, i * 3:(i + 1) * 3] - 1, -1,
                                                      1)))
        tf.summary.histogram("tx", self.pred_poses[:, :, 0])
        tf.summary.histogram("ty", self.pred_poses[:, :, 1])
        tf.summary.histogram("tz", self.pred_poses[:, :, 2])
        tf.summary.histogram("rx", self.pred_poses[:, :, 3])
        tf.summary.histogram("ry", self.pred_poses[:, :, 4])
        tf.summary.histogram("rz", self.pred_poses[:, :, 5])
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name + "/values", var)
        # for grad, var in self.grads_and_vars:
        #     tf.summary.histogram(var.op.name + "/gradients", grad)


    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.) / 2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)


def model_train_all():
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if len(os.listdir(FLAGS.checkpoint_dir)) != 0:
        for file in os.listdir(FLAGS.checkpoint_dir):
            os.remove(os.path.join(FLAGS.checkpoint_dir, file))
    train()


def model_test_depth():
    ckpt_name = find_latest_ckpt(FLAGS.checkpoint_dir)
    ckpt_abs_file_path = os.path.join(FLAGS.checkpoint_dir, ckpt_name)
    with open(FLAGS.test_file_list, 'r') as f:
        test_files = f.readlines()
        test_files = [FLAGS.dataset_dir + t[:-1] for t in test_files]
    basename = os.path.basename(ckpt_name)


    input_image,pred_depth = build_depth_test_graph()
    fetches = {}
    fetches['depth'] = pred_depth

    saver = tf.train.Saver([var for var in tf.trainable_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt_abs_file_path)
        pred_all = []
        for t in range(0, len(test_files), FLAGS.batch_size):
            if t % 100 == 0:
                print('processing %s: %d/%d' % (basename, t, len(test_files)))
            inputs = np.zeros(
                (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3),
                dtype=np.uint8)
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                fh = open(test_files[idx], 'rb')  # to fix the UnicodeDecodeError about utf8,change 'r' to 'rb'
                raw_im = pil.open(fh)
                scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)
                inputs[b] = np.array(scaled_im)
                # im = scipy.misc.imread(test_files[idx])
                # inputs[b] = scipy.misc.imresize(im, (FLAGS.img_height, FLAGS.img_width))
            pred = sess.run(fetches, feed_dict={input_image: inputs})
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                pred_all.append(pred['depth'][b, :, :, 0])
        evaluate_depth(pred_all, FLAGS.test_file_list,
                       FLAGS.dataset_dir,
                       FLAGS.min_depth,
                       FLAGS.max_depth)
    pass


def model_test_pose():
    """
    测试pose
    1. 根据要测试的sequence_id,读取数据集,找到对应的pose_Groundtruth
    2. 把原有的12参数变更为8参数,存储成all.txt
    3. 按照sequence_length,把上述数据存储成一系列文件.
    4. 连接图片,进入网络进行测试
    5. 把测试结果存储成一系列文件
    6. 对比这些文件,得到测试结果
    这里需要注意的是,每次我们输入sequence_length张图片进入网络,网络会把第一张图作为初始位置,
    但是在GT中,整个序列上千张图片都是连续的,所以直接比较某张图的pose是没有意义的
    比如输入345这三张图,得到p1,p2,p3,
    在GT中,对应的三张图的pose是P1,P2,P3,
    我们要做的其实是:计算p1,p2之间的运动,和P1,P2之间的情况作比较
    :return:
    """
    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    input_image, pred_poses = build_pose_test_graph()
    fetches = {}
    fetches['pose'] = pred_poses

    # 把原有的12参数变更为8参数,存储成all.txt
    seq_dir = os.path.join(FLAGS.dataset_dir, 'sequences', '%.2d' % FLAGS.test_seq)
    img_dir = os.path.join(seq_dir, 'image_2')
    N = len(glob(img_dir + '/*.png'))  # 这样直接读取的文件名是乱序的,我们只用文件数量
    test_frames = ['%.2d %.6d' % (FLAGS.test_seq, n) for n in range(N)]  # 000000~N,生成按顺序的图片文件名列表

    # 读取测试序列对应的times
    with open(FLAGS.dataset_dir + 'sequences/%.2d/times.txt' % FLAGS.test_seq, 'r') as f:
        times_strs = f.readlines()
        times = np.array([float(s[:-1]) for s in times_strs])

    output_file_name = FLAGS.output_dir + '/%.2d' % FLAGS.test_seq + "_pose_gt_all.txt"
    pose_gt_file = os.path.join(FLAGS.dataset_dir, 'poses', '%.2d.txt' % FLAGS.test_seq)

    # 创建8参数ground truth 列表 尺寸[N][8],N为该序列的图片数,8依次为:
    # [时间 Tx Ty Tz Qx Qy Qz Qw],其中(Tx Ty Tz)是表示平移的向量,(Qx Qy Qz Qw)是表示旋转的四元数
    GT_8params_list = Odometry_12params_to_8params(pose_gt_file, times, output_file_name)

    groundTruthPath = FLAGS.output_dir + "/GroundTruth/"
    if not os.path.isdir(groundTruthPath):
        os.makedirs(groundTruthPath)

    create_8params_gtfiles(output_file_name, groundTruthPath, FLAGS.seq_length)

    max_src_offset = (FLAGS.seq_length - 1) // 2
    ckpt_name = find_latest_ckpt(FLAGS.checkpoint_dir)
    ckpt_abs_file_path = os.path.join(FLAGS.checkpoint_dir, ckpt_name)
    saver = tf.train.Saver([var for var in tf.trainable_variables()])
    with tf.Session() as sess:
        saver.restore(sess, ckpt_abs_file_path)
        for tgt_idx in range(N):
            if not is_valid_sample(test_frames, tgt_idx, FLAGS.seq_length):
                continue
            if tgt_idx % 100 == 0:
                print('Progress: %d/%d' % (tgt_idx, N))
            # TODO: currently assuming batch_size = 1
            image_seq = load_image_sequence(FLAGS.dataset_dir,
                                            test_frames,
                                            tgt_idx,
                                            FLAGS.seq_length,
                                            FLAGS.img_height,
                                            FLAGS.img_width)

            pred = sess.run(fetches, feed_dict={input_image: image_seq[None, :, :, :]})
            pred_poses = pred['pose'][0]
            # Insert the target pose [0, 0, 0, 0, 0, 0]
            pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1, 6)), axis=0)
            curr_times = times[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]

            predictionPath = FLAGS.output_dir + "/Prediction/"
            if not os.path.isdir(predictionPath):
                os.makedirs(predictionPath)

            out_file = predictionPath + '/%.6d.txt' % (tgt_idx - max_src_offset)

            predictions_list = []  # [N,FLAGS.seq_length,8],N为测试序列的图片数量,
                                   # FLAGS.seq_length为一个样本中有几张图片
                                   # 8为每条数据的8个参数
            single_sample = []     # [FLAGS.seq_length,8],单个样本

            #dump_pose_seq_TUM(out_file, pred_poses, curr_times)
            # First frame as the origin
            #(out_file, poses, times)
            first_pose = pose_vec_to_mat(pred_poses[0])
            with open(out_file, 'w') as f:
                for p in range(len(curr_times)):
                    single_sample = []
                    this_pose = pose_vec_to_mat(pred_poses[p])
                    this_pose = np.dot(first_pose, np.linalg.inv(this_pose))
                    tx = this_pose[0, 3]
                    ty = this_pose[1, 3]
                    tz = this_pose[2, 3]
                    rot = this_pose[:3, :3]
                    qw, qx, qy, qz = rot2quat(rot)
                    single_sample.append([curr_times[p], tx, ty, tz, qx, qy, qz, qw])
                    f.write('%f %f %f %f %f %f %f %f\n' % (curr_times[p], tx, ty, tz, qx, qy, qz, qw))
                predictions_list.append(single_sample)
    # evaluate_pose(predictionPath, groundTruthPath)

    pred_files = glob(predictionPath + '/*.txt')
    ate_all = []

    # 从GT_8params_list中取出对应的FLAGS.seq_length条数据
    # 读取对应的predictions_list中的FLAGS.seq_length条数据
    # 把上述两种数据都转换为字典,key是时间戳,value是后面七个值
    # 传入compute_ate2进行计算
    for i in range(len(predictions_list)):
        gt_sample = {}
        pred_sample = {}
        for j in range(FLAGS.seq_length):
            gt_sample[GT_8params_list[i+j][0]] = GT_8params_list[i+j][1:]
            pred_sample[predictions_list[i][j][0]] = predictions_list[i][j][1:]
        ate = compute_ate2(gt_sample, pred_sample)
        if ate == False:
            continue
        ate_all.append(ate)
        pass


    # for i in range(len(pred_files)):
    #     gtruth_file = groundTruthPath + os.path.basename(pred_files[i])
    #     if not os.path.exists(gtruth_file):
    #         continue
    #     ate = compute_ate(gtruth_file, pred_files[i])
    #     if ate == False:
    #         continue
    #     ate_all.append(ate)
    ate_all = np.array(ate_all)
    print("Predictions dir: %s" % predictionPath)
    print("ATE(Absolute Trajectory Error,绝对轨迹误差) mean: %.4f, std: %.4f" % (np.mean(ate_all), np.std(ate_all)))

    pass


def main(_):
    start_time = time.time()
    if FLAGS.run_mode == 0:
        model_train_all()
    elif FLAGS.run_mode == 1:
        model_test_depth()
    elif FLAGS.run_mode == 2:
        model_test_pose()
    time_elapsed = time.time() - start_time
    h = time_elapsed // 3600
    m = (time_elapsed - 3600 * h) // 60
    s = time_elapsed - 3600 * h - m * 60
    print('Complete in {:.0f}h {:.0f}m {:.0f}s'.format(h, m, s))
    pass


if __name__ == '__main__':
    tf.app.run()
   # print(get_available_gpus())
