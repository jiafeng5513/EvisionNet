# -*- coding: utf-8 -*-
from __future__ import division
import time
import math
import tensorflow.contrib.slim as slim
import pprint
import random
import PIL.Image as pil
import scipy.misc
from pose_evaluation_utils import dump_pose_seq_TUM
from tensorflow.contrib.layers.python.layers import utils
from glob import glob
from data_loader import DataLoader
from utils import *

# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.01

flags = tf.app.flags
flags.DEFINE_integer("run_mode", 2, "0=train,1=test_depth,2=test_pose")

flags.DEFINE_string("dataset_dir", "/home/RAID1/DataSet/KITTI/KittiOdometry/", "数据位置")
# KittiOdometry,KittiOdometry_prepared,KittiRaw,KittiRaw_prepared

# test pose  : KittiOdometry
# test depth : KittiRaw
# train      : KittiRaw_prepared

flags.DEFINE_string("checkpoint_dir", "../checkpoints/", "用于保存和加载ckpt的目录")

# params for model_train
flags.DEFINE_string("init_checkpoint_file", None, "用来初始化的ckpt")
flags.DEFINE_float("learning_rate", 0.0002, "学习率")
flags.DEFINE_float("beta1", 0.9, "adam动量参数")
flags.DEFINE_float("smooth_weight", 0.5, "平滑的权重")
flags.DEFINE_float("explain_reg_weight", 0.0, "Weight for explanability regularization")
flags.DEFINE_integer("batch_size", 1, "batch size")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length", 3, "一个样本中含有几张图片")
flags.DEFINE_integer("max_steps", 200000, "训练迭代次数")
flags.DEFINE_integer("summary_freq", 100, "summary频率")
flags.DEFINE_integer("save_latest_freq", 5000,"保存最新模型的频率(会覆盖之前保存的最新模型)")
flags.DEFINE_boolean("continue_train", False, "是否从之前的ckpt继续训练")

# params for model_test_depth
flags.DEFINE_string("test_file_list",'../data/kitti/test_files_eigen.txt',"Path to the list of test files")
flags.DEFINE_float("min_depth",1e-3,"Threshold for minimum depth")
flags.DEFINE_float("max_depth",80,"Threshold for maximum depth")

# params for model_test_pose
flags.DEFINE_integer("test_seq", 9, "使用KittiOdometry的哪个序列进行测试")  # pick from 22 sequences in KittiOdometry
flags.DEFINE_string("output_dir", "./test_output/test_pose/", "Output directory")

# add by jiafeng5513,followed by https://github.com/tinghuiz/SfMLearner/pull/70
flags.DEFINE_integer("num_source", None, "number of source images")
flags.DEFINE_integer("num_scales", None, "number of used image scales")

FLAGS = flags.FLAGS

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])


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


def pose_exp_net(tgt_image, src_image_stack, do_exp=True, is_training=True):
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    num_source = int(src_image_stack.get_shape()[3].value // 3)
    with tf.variable_scope('pose_exp_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1 = slim.conv2d(inputs, 16, [7, 7], stride=2, scope='cnv1')
            cnv2 = slim.conv2d(cnv1, 32, [5, 5], stride=2, scope='cnv2')
            cnv3 = slim.conv2d(cnv2, 64, [3, 3], stride=2, scope='cnv3')
            cnv4 = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5 = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            # Pose specific layers
            with tf.variable_scope('pose'):
                cnv6 = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                cnv7 = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cnv7, 6 * num_source, [1, 1], scope='pred',
                                        stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant
                # facilitates training.
                pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
            # Exp mask specific layers
            if do_exp:
                with tf.variable_scope('exp'):
                    upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

                    upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
                    mask4 = slim.conv2d(upcnv4, num_source * 2, [3, 3], stride=1, scope='mask4',
                                        normalizer_fn=None, activation_fn=None)

                    upcnv3 = slim.conv2d_transpose(upcnv4, 64, [3, 3], stride=2, scope='upcnv3')
                    mask3 = slim.conv2d(upcnv3, num_source * 2, [3, 3], stride=1, scope='mask3',
                                        normalizer_fn=None, activation_fn=None)

                    upcnv2 = slim.conv2d_transpose(upcnv3, 32, [5, 5], stride=2, scope='upcnv2')
                    mask2 = slim.conv2d(upcnv2, num_source * 2, [5, 5], stride=1, scope='mask2',
                                        normalizer_fn=None, activation_fn=None)

                    upcnv1 = slim.conv2d_transpose(upcnv2, 16, [7, 7], stride=2, scope='upcnv1')
                    mask1 = slim.conv2d(upcnv1, num_source * 2, [7, 7], stride=1, scope='mask1',
                                        normalizer_fn=None, activation_fn=None)
            else:
                mask1 = None
                mask2 = None
                mask3 = None
                mask4 = None
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return pose_final, [mask1, mask2, mask3, mask4], end_points


def disp_net(tgt_image, is_training=True):
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1 = slim.conv2d(tgt_image, 32, [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1, 32, [7, 7], stride=1, scope='cnv1b')
            cnv2 = slim.conv2d(cnv1b, 64, [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2, 64, [5, 5], stride=1, scope='cnv2b')
            cnv3 = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3, 128, [3, 3], stride=1, scope='cnv3b')
            cnv4 = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4, 256, [3, 3], stride=1, scope='cnv4b')
            cnv5 = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5, 512, [3, 3], stride=1, scope='cnv5b')
            cnv6 = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6, 512, [3, 3], stride=1, scope='cnv6b')
            cnv7 = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7, 512, [3, 3], stride=1, scope='cnv7b')

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7 = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6 = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5 = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            i4_in = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4 = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4 = DISP_SCALING * slim.conv2d(icnv4, 1, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H / 4), np.int(W / 4)])

            upcnv3 = slim.conv2d_transpose(icnv4, 64, [3, 3], stride=2, scope='upcnv3')
            i3_in = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3 = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')
            disp3 = DISP_SCALING * slim.conv2d(icnv3, 1, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H / 2), np.int(W / 2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 32, [3, 3], stride=2, scope='upcnv2')
            i2_in = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2 = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')
            disp2 = DISP_SCALING * slim.conv2d(icnv2, 1, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, 16, [3, 3], stride=2, scope='upcnv1')
            i1_in = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1 = slim.conv2d(i1_in, 16, [3, 3], stride=1, scope='icnv1')
            disp1 = DISP_SCALING * slim.conv2d(icnv1, 1, [3, 3], stride=1,
                                               activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP

            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1, disp2, disp3, disp4], end_points


def load_image_sequence(dataset_dir, frames,  tgt_idx, seq_length, img_height, img_width):
    half_offset = int((seq_length - 1)/2)
    for o in range(-half_offset, half_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        img_file = os.path.join(
            dataset_dir, 'sequences', '%s/image_2/%s.png' % (curr_drive, curr_frame_id))
        curr_img = scipy.misc.imread(img_file)
        curr_img = scipy.misc.imresize(curr_img, (img_height, img_width))
        if o == -half_offset:
            image_seq = curr_img
        else:
            image_seq = np.hstack((image_seq, curr_img))
    return image_seq


def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)
    tgt_drive, _ = frames[tgt_idx].split(' ')
    max_src_offset = int((seq_length - 1)/2)
    min_src_idx = tgt_idx - max_src_offset
    max_src_idx = tgt_idx + max_src_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    # TODO: unnecessary to check if the drives match
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False


class EvisionNet(object):
    def __init__(self):
        pass
    
    def build_train_graph(self):
        opt = self.opt
        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales)
        with tf.name_scope("data_loading"):
            tgt_image, src_image_stack, intrinsics = loader.load_train_batch()
            tgt_image = self.preprocess_image(tgt_image)
            src_image_stack = self.preprocess_image(src_image_stack)

        with tf.name_scope("depth_prediction"):
            pred_disp, depth_net_endpoints = disp_net(tgt_image, is_training=True)
            pred_depth = [1./d for d in pred_disp]

        with tf.name_scope("pose_and_explainability_prediction"):  # , pose_exp_net_endpoints
            pred_poses, pred_exp_logits, pose_exp_net_endpoints = pose_exp_net(tgt_image,src_image_stack,
                                                        do_exp=(opt.explain_reg_weight > 0),
                                                        is_training=True)

        with tf.name_scope("compute_loss"):
            pixel_loss = 0
            exp_loss = 0
            smooth_loss = 0
            tgt_image_all = []
            src_image_stack_all = []
            proj_image_stack_all = []
            proj_error_stack_all = []
            exp_mask_stack_all = []
            for s in range(opt.num_scales):
                if opt.explain_reg_weight > 0:
                    # Construct a reference explainability mask (i.e. all 
                    # pixels are explainable)
                    ref_exp_mask = self.get_reference_explain_mask(s)
                # Scale the source and target images for computing loss at the 
                # according scale.
                curr_tgt_image = tf.image.resize_area(tgt_image, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])                
                curr_src_image_stack = tf.image.resize_area(src_image_stack, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])

                if opt.smooth_weight > 0:
                    smooth_loss += opt.smooth_weight/(2**s) * \
                        self.compute_smooth_loss(pred_disp[s])

                for i in range(opt.num_source):
                    # Inverse warp the source image to the target image frame
                    curr_proj_image = projective_inverse_warp(
                        curr_src_image_stack[:,:,:,3*i:3*(i+1)], 
                        tf.squeeze(pred_depth[s], axis=3), 
                        pred_poses[:,i,:], 
                        intrinsics[:,s,:,:])
                    curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)
                    # Cross-entropy loss as regularization for the 
                    # explainability prediction
                    if opt.explain_reg_weight > 0:
                        curr_exp_logits = tf.slice(pred_exp_logits[s], 
                                                   [0, 0, 0, i*2], 
                                                   [-1, -1, -1, 2])
                        exp_loss += opt.explain_reg_weight * \
                            self.compute_exp_reg_loss(curr_exp_logits,
                                                      ref_exp_mask)
                        curr_exp = tf.nn.softmax(curr_exp_logits)
                    # Photo-consistency loss weighted by explainability
                    if opt.explain_reg_weight > 0:
                        pixel_loss += tf.reduce_mean(curr_proj_error * \
                            tf.expand_dims(curr_exp[:,:,:,1], -1))
                    else:
                        pixel_loss += tf.reduce_mean(curr_proj_error) 
                    # Prepare images for tensorboard summaries
                    if i == 0:
                        proj_image_stack = curr_proj_image
                        proj_error_stack = curr_proj_error
                        if opt.explain_reg_weight > 0:
                            exp_mask_stack = tf.expand_dims(curr_exp[:,:,:,1], -1)
                    else:
                        proj_image_stack = tf.concat([proj_image_stack, 
                                                      curr_proj_image], axis=3)
                        proj_error_stack = tf.concat([proj_error_stack, 
                                                      curr_proj_error], axis=3)
                        if opt.explain_reg_weight > 0:
                            exp_mask_stack = tf.concat([exp_mask_stack, 
                                tf.expand_dims(curr_exp[:,:,:,1], -1)], axis=3)
                tgt_image_all.append(curr_tgt_image)
                src_image_stack_all.append(curr_src_image_stack)
                proj_image_stack_all.append(proj_image_stack)
                proj_error_stack_all.append(proj_error_stack)
                if opt.explain_reg_weight > 0:
                    exp_mask_stack_all.append(exp_mask_stack)
            total_loss = pixel_loss + smooth_loss + exp_loss

        with tf.name_scope("train_op"):
            train_vars = [var for var in tf.trainable_variables()]
            optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)
            # self.grads_and_vars = optim.compute_gradients(total_loss, 
            #                                               var_list=train_vars)
            # self.train_op = optim.apply_gradients(self.grads_and_vars)
            self.train_op = slim.learning.create_train_op(total_loss, optim)
            self.global_step = tf.Variable(0, 
                                           name='global_step', 
                                           trainable=False)
            self.incr_global_step = tf.assign(self.global_step, 
                                              self.global_step+1)

        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_depth = pred_depth
        self.pred_poses = pred_poses
        self.steps_per_epoch = loader.steps_per_epoch
        self.total_loss = total_loss
        self.pixel_loss = pixel_loss
        self.exp_loss = exp_loss
        self.smooth_loss = smooth_loss
        self.tgt_image_all = tgt_image_all
        self.src_image_stack_all = src_image_stack_all
        self.proj_image_stack_all = proj_image_stack_all
        self.proj_error_stack_all = proj_error_stack_all
        self.exp_mask_stack_all = exp_mask_stack_all

    def get_reference_explain_mask(self, downscaling):
        opt = self.opt
        tmp = np.array([0,1])
        ref_exp_mask = np.tile(tmp, 
                               (opt.batch_size, 
                                int(opt.img_height/(2**downscaling)), 
                                int(opt.img_width/(2**downscaling)), 
                                1))
        ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
        return ref_exp_mask

    def compute_exp_reg_loss(self, pred, ref):
        l = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(ref, [-1, 2]),
            logits=tf.reshape(pred, [-1, 2]))
        return tf.reduce_mean(l)

    def compute_smooth_loss(self, pred_disp):
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

    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_loss", self.pixel_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        tf.summary.scalar("exp_loss", self.exp_loss)
        for s in range(opt.num_scales):
            tf.summary.histogram("scale%d_depth" % s, self.pred_depth[s])
            tf.summary.image('scale%d_disparity_image' % s, 1./self.pred_depth[s])
            tf.summary.image('scale%d_target_image' % s, \
                             self.deprocess_image(self.tgt_image_all[s]))
            for i in range(opt.num_source):
                if opt.explain_reg_weight > 0:
                    tf.summary.image(
                        'scale%d_exp_mask_%d' % (s, i), 
                        tf.expand_dims(self.exp_mask_stack_all[s][:,:,:,i], -1))
                tf.summary.image(
                    'scale%d_source_image_%d' % (s, i), 
                    self.deprocess_image(self.src_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
                tf.summary.image('scale%d_projected_image_%d' % (s, i), 
                    self.deprocess_image(self.proj_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
                tf.summary.image('scale%d_proj_error_%d' % (s, i),
                    self.deprocess_image(tf.clip_by_value(self.proj_error_stack_all[s][:,:,:,i*3:(i+1)*3] - 1, -1, 1)))
        tf.summary.histogram("tx", self.pred_poses[:,:,0])
        tf.summary.histogram("ty", self.pred_poses[:,:,1])
        tf.summary.histogram("tz", self.pred_poses[:,:,2])
        tf.summary.histogram("rx", self.pred_poses[:,:,3])
        tf.summary.histogram("ry", self.pred_poses[:,:,4])
        tf.summary.histogram("rz", self.pred_poses[:,:,5])
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name + "/values", var)
        # for grad, var in self.grads_and_vars:
        #     tf.summary.histogram(var.op.name + "/gradients", grad)

    def train(self, opt):
        opt.num_source = opt.seq_length - 1
        # TODO: currently fixed to 4
        opt.num_scales = 4
        self.opt = opt
        self.build_train_graph()
        self.collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                    [self.global_step],
                                     max_to_keep=10)
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, 
                                 save_summaries_secs=0, 
                                 saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)
            start_time = time.time()
            for step in range(1, opt.max_steps):
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }

                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["summary"] = sv.summary_op

                results = sess.run(fetches)
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
                            % (train_epoch, train_step, self.steps_per_epoch, \
                                (time.time() - start_time)/opt.summary_freq, 
                                results["loss"]))
                    start_time = time.time()

                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

                if step % self.steps_per_epoch == 0:
                    self.save(sess, opt.checkpoint_dir, gs)

    def build_depth_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
                    self.img_height, self.img_width, 3], name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        with tf.name_scope("depth_prediction"):
            pred_disp, depth_net_endpoints = disp_net(
                input_mc, is_training=False)
            pred_depth = [1./disp for disp in pred_disp]
        pred_depth = pred_depth[0]
        self.inputs = input_uint8
        self.pred_depth = pred_depth
        self.depth_epts = depth_net_endpoints

    def build_pose_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, self.img_height, self.img_width * self.seq_length, 3],
                                     name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)
        with tf.name_scope("pose_prediction"):
            pred_poses, _, _ = pose_exp_net(tgt_image, src_image_stack, do_exp=False, is_training=False)
            self.inputs = input_uint8
            self.pred_poses = pred_poses

    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. -1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def setup_inference(self, img_height, img_width, mode, seq_length=3, batch_size=1):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'depth':
            self.build_depth_test_graph()
        if self.mode == 'pose':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph()

    def inference(self, inputs, sess, mode='depth'):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)


def model_train_all():
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    sfm = EvisionNet()
    sfm.train(FLAGS)


def model_test_depth():
    ckpt_name = find_latest_ckpt(FLAGS.checkpoint_dir)
    ckpt_abs_file_path = os.path.join(FLAGS.checkpoint_dir,ckpt_name)
    with open(FLAGS.test_file_list, 'r') as f:
        test_files = f.readlines()
        test_files = [FLAGS.dataset_dir + t[:-1] for t in test_files]
    basename = os.path.basename(ckpt_name)
    sfm = EvisionNet()
    sfm.setup_inference(img_height=FLAGS.img_height,
                        img_width=FLAGS.img_width,
                        batch_size=FLAGS.batch_size,
                        mode='depth')
    saver = tf.train.Saver([var for var in tf.model_variables()])
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
                fh = open(test_files[idx],'rb')  # to fix the UnicodeDecodeError about utf8,change 'r' to 'rb'
                raw_im = pil.open(fh)
                scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)
                inputs[b] = np.array(scaled_im)
                # im = scipy.misc.imread(test_files[idx])
                # inputs[b] = scipy.misc.imresize(im, (FLAGS.img_height, FLAGS.img_width))
            pred = sfm.inference(inputs, sess, mode='depth')
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                pred_all.append(pred['depth'][b, :, :, 0])
        evaluate_depth(pred_all,FLAGS.test_file_list,
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
    :return:
    """
    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    pose_gt_file = os.path.join(FLAGS.dataset_dir,'poses','%.2d.txt' % FLAGS.test_seq)

    # 把原有的12参数变更为8参数,存储成all.txt

    sfm = EvisionNet()
    sfm.setup_inference(img_height=FLAGS.img_height,
                        img_width=FLAGS.img_width,
                        mode='pose',
                        seq_length=FLAGS.seq_length)
    saver = tf.train.Saver([var for var in tf.trainable_variables()])


    seq_dir = os.path.join(FLAGS.dataset_dir, 'sequences', '%.2d' % FLAGS.test_seq)
    img_dir = os.path.join(seq_dir, 'image_2')
    N = len(glob(img_dir + '/*.png'))
    test_frames = ['%.2d %.6d' % (FLAGS.test_seq, n) for n in range(N)]  # 000000~N

    with open(FLAGS.dataset_dir + 'sequences/%.2d/times.txt' % FLAGS.test_seq, 'r') as f:
        times = f.readlines()

    times = np.array([float(s[:-1]) for s in times])
    output_file_name = FLAGS.output_dir +'/%.2d' % FLAGS.test_seq+"_pose_gt_all.txt"

    Odometry_12params_to_8params(pose_gt_file, times, output_file_name)
    groundTruthPath = FLAGS.output_dir+"/GroundTruth/"
    if not os.path.isdir(groundTruthPath):
        os.makedirs(groundTruthPath)

    create_8params_gtfiles(output_file_name, groundTruthPath, FLAGS.seq_length)


    max_src_offset = (FLAGS.seq_length - 1) // 2
    ckpt_name = find_latest_ckpt(FLAGS.checkpoint_dir)
    ckpt_abs_file_path = os.path.join(FLAGS.checkpoint_dir,ckpt_name)
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
            pred = sfm.inference(image_seq[None, :, :, :], sess, mode='pose')
            pred_poses = pred['pose'][0]
            # Insert the target pose [0, 0, 0, 0, 0, 0]
            pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1, 6)), axis=0)
            curr_times = times[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]

            predictionPath = FLAGS.output_dir + "/Prediction/"
            if not os.path.isdir(predictionPath):
                os.makedirs(predictionPath)

            out_file = predictionPath + '/%.6d.txt' % (tgt_idx - max_src_offset)

            dump_pose_seq_TUM(out_file, pred_poses, curr_times)
    evaluate_pose(predictionPath, groundTruthPath)
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
    m = (time_elapsed - 3600*h) // 60
    s = time_elapsed - 3600*h - m*60
    print('Complete in {:.0f}h {:.0f}m {:.0f}s'.format(h,m,s))
    pass


if __name__ == '__main__':
    tf.app.run()