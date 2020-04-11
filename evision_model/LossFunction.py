# -*- coding: utf-8 -*-
"""
Loss Function for EvisionNet

based on "Depth from video in the wild", "SfmLearner-PyTorch", "SfmLearner-TF" and "struct2depth"

Total Loss = Depth Smoothing +          # 2. 深度平滑

Reconstr loss +            # 1. 重投影损失

             ssim loss +                # 3. structural similarity index,结构相似性损失

             Depth Consistency loss +   # 5.
             Rot_loss +                 # 6. 旋转损失
             Trans_loss                 # 7. 平移损失
             Motion Smoothing +         # 4. 运动平滑(需要seg_stack即object_mask)
            (以上各项均带有权重系数超参数)
code by jiafeng5513

NOTE:
    1. TensorFlow 的默认顺序是 [N H W C], PyTorch的默认顺序是 [N C H W]
    2. 计算损失的最小单位是一个batch,一个batch含有batch size 个 SEQ_LENGTH长度的图片单元,
       每个图片单元是预处理时生成的一张连体照片由SEQ_LENGTH个连续帧组成
    3. image的联结方式是在C维度,也就是通道上进行连接
    4. 由于2,每次计算损失时,关于深度的损失有SEQ_LENGTH个,关于ego-motion的损失有SEQ_LENGTH-1个
    5. 每相邻的两张image进MotionNet,然后再调换这两张image的顺序.
        假设SEQ_LENGTH=5,那么MotionNet需要这样计算: 12/21,23/32,34/43,45/54,共(SEQ_LENGTH-1)*2次
        其中正序的一组存储在rotation_pred,translation_pred,residual_translation_pred,intrinsic_pred中
        逆序的一组存储在inv_rotation_pred,inv_translation_pred,inv_residual_translation_pred,inv_intrinsic_pred中
    6.

"""
import torch
import torch.nn as nn
from evision_model import transform_depth_map


# part 1 :深度平滑
def Depth_Smoothing_Loss(depth_pred, SEQ_LENGTH, image_stack):
    """
    total loss part 1 :深度平滑
    TODO:这里实际上计算的是视差的平滑
    Args:
        image_stack:输入图像,在通道维度上层叠,[b,3*SEQ_LENGTH,h,w]
        SEQ_LENGTH:序列长度       integer like 3 or 5
        depth_pred:深度预测       list of [b,1,h,w], list length = SEQ_LENGTH
    Returns:
        Depth_Smoothing
    """

    smooth_loss = 0
    disp = {}
    for i in range(SEQ_LENGTH):
        disp[i] = 1.0 / depth_pred.depth[i]

    for i in range(SEQ_LENGTH):
        disp_smoothing = disp[i]
        # Perform depth normalization, dividing by the mean.
        mean_disp = torch.mean(disp_smoothing, axis=[1, 2, 3], keep_dims=True)
        disp_input = disp_smoothing / mean_disp
        smooth_loss += _depth_smoothness(disp_input, image_stack[:, 3 * i:3 * (i + 1), :, :])
    return smooth_loss
    pass

"""
训练现场:
1. train函数每次处理的是一个epoch,在内部实际上的逐个batch进行反向传播的
2. 对于每个batch,我们需要让depthnet前向传播SEQ_LENGTH次,让motion_net前向传播(SEQ_LENGTH-1)*2次
假设SEQ_LENGTH=5,那么MotionNet需要这样计算: 12/21,23/32,34/43,45/54,共(SEQ_LENGTH-1)*2次
        其中正序的一组存储在rotation_pred,translation_pred,residual_translation_pred,intrinsic_pred中
        逆序的一组存储在inv_rotation_pred,inv_translation_pred,inv_residual_translation_pred,inv_intrinsic_pred中
        这8个list的长度都是SEQ_LENGTH-1
"""

# part N: 重投影损失
def ReprojectLoss():

    pass







def getTotalLoss(image_stack,                     # 训练图片  SEQ_LENGTH张图片在通道维度(即第二维度上)层叠
                 depth_pred,                      # 深度预测       list of [b,1,h,w], list length = SEQ_LENGTH
                 rotation_pred,                   # 正向-旋转预测  list of [b,3],     list length = SEQ_LENGTH-1
                 inv_rotation_pred,               # 逆序-旋转预测  list of [b,3],     list length = SEQ_LENGTH-1
                 translation_pred,                # 正向-平移预测  list of [b,3,1,1], list length = SEQ_LENGTH-1
                 inv_translation_pred,            # 逆序-平移预测  list of [b,3,1,1], list length = SEQ_LENGTH-1
                 residual_translation_pred,       # 正向-场预测    list of [b,3,h,w], list length = SEQ_LENGTH-1
                 inv_residual_translation_pred,   # 逆序-场预测    list of [b,3,h,w], list length = SEQ_LENGTH-1
                 intrinsic_pred,                  # 正向-内参预测  list of [b,3,3],   list length = SEQ_LENGTH-1
                 inv_intrinsic_pred,              # 逆序-内参预测  list of [b,3,3],   list length = SEQ_LENGTH-1

                 GT_intrinsic_mat,           # GT内参
                 smooth_weight,              # 平滑损失权重
                 learn_intrinsics=True,      # 是否学习内参预测
                 foreground_dilation=8,      #
                 SEQ_LENGTH = 3              # 单位训练序列长度,默认为3
                 ):
    # 第一部分:disp_smoothing
    reconstr_loss = 0
    smooth_loss = 0
    ssim_loss = 0
    depth_consistency_loss = 0
    disp = {}
    for i in range(SEQ_LENGTH):
        disp[i] = 1.0 / depth_pred.depth[i]

    if smooth_weight > 0:
        for i in range(SEQ_LENGTH):
            disp_smoothing = disp[i]
            # Perform depth normalization, dividing by the mean.
            mean_disp = torch.mean(disp_smoothing, axis=[1, 2, 3], keep_dims=True)
            disp_input = disp_smoothing / mean_disp
            smooth_loss += _depth_smoothness(disp_input, image_stack[:, 3 * i:3 * (i + 1), :, :])

    rot_loss = 0.0
    trans_loss = 0.0
    # 第二部分:motion_smoothing
    motion_smoothing = 0.0
    """
    每次损失函数计算时,涉及到SEQ_LENGTH个DepthNet的输出:
    depth_pred
    以及SEQ_LENGTH - 1个MotionNet的输出:
    rotation, translation, residual_translation, intrinsic_mat
    """
    for i in range(SEQ_LENGTH - 1):
        j = i + 1
        depth_i = depth_pred[i][:, 0, :, :]
        depth_j = depth_pred[j][:, 0, :, :]
        image_j = image_stack[:, 3 * j:3 * (j + 1), :, :]
        image_i = image_stack[:, i * 3:(i + 1) * 3, :, :]


    # We feed the pair of images into the network, once in forward order and then in reverse order.
    # The results are fed into the loss calculation.
    # The following losses are calculated:
    # - RGB and SSIM (structural similarity index) photometric consistency.
    # - Cycle consistency of rotations and translations for every pixel.
    # - L1 smoothness of the disparity and the motion field.
    # - Depth consistency

    rot, trans, trans_res, mat = motion_prediction_net.motion_field_net(
        images=tf.concat([image_i, image_j], axis=-1),weight_reg=self.weight_reg)
    inv_rot, inv_trans, inv_trans_res, inv_mat = (
     motion_prediction_net.motion_field_net(images=tf.concat([image_j, image_i], axis=-1),weight_reg=self.weight_reg))
    if learn_intrinsics:
        intrinsic_mat = 0.5 * (mat + inv_mat)
    else:
        intrinsic_mat = GT_intrinsic_mat[:, 0, :, :]

    motion_smoothing += _smoothness(trans_res)
    motion_smoothing += _smoothness(inv_trans_res)

    transformed_depth_j = transform_depth_map.using_motion_vector(depth_pred, trans, rot, intrinsic_mat)
    pass


def _depth_smoothness(depth, img):
    """Computes image-aware depth smoothness loss."""
    depth_dx = _gradient_x(depth)
    depth_dy = _gradient_y(depth)
    image_dx = _gradient_x(img)
    image_dy = _gradient_y(img)
    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), 1, keepdims=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), 1, keepdims=True))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return torch.mean(abs(smoothness_x)) + torch.mean(abs(smoothness_y))


def _gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def _gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def dilate(x, foreground_dilation):
    # Dilation by n pixels is roughtly max pooling by 2 * n + 1.
    p = foreground_dilation * 2 + 1
    pool2d = nn.MaxPool2d((p, p), stride=(1, 1))  # TODO:tf.nn.max_pool(x, [1, p, p, 1], [1] * 4, 'SAME')
    return pool2d(x)


# 很难做出来,我们需要先搞定数据处理,然后看看原来的代码里面这一部分究竟做了什么,然后再写等效代码
def get_object_masks(batch_size, seg_stack, SEQ_LENGTH):
    object_masks = []
    for i in range(batch_size):
        object_ids = torch.unique(seg_stack[i].reshape([-1]))[0]
        object_masks_i = []
        for j in range(SEQ_LENGTH):
            current_seg = seg_stack[i, j * 3, :, :]  # (H, W)

            def process_obj_mask(obj_id):
                """Create a mask for obj_id, skipping the background mask."""
                mask = tf.logical_and(
                    torch.eq(current_seg, obj_id),  # pylint: disable=cell-var-from-loop
                    ~torch.eq(tf.cast(0, tf.uint8), obj_id))
                # Leave out vert small masks, that are most often errors.
                size = tf.reduce_sum(tf.to_int32(mask))
                mask = tf.logical_and(mask, tf.greater(size, MIN_OBJECT_AREA))
                if not self.boxify:
                    return mask
                # Complete the mask to its bounding box.
                binary_obj_masks_y = tf.reduce_any(mask, axis=1, keepdims=True)
                binary_obj_masks_x = tf.reduce_any(mask, axis=0, keepdims=True)
                return tf.logical_and(binary_obj_masks_y, binary_obj_masks_x)

            object_mask = tf.map_fn(  # (N, H, W)
                process_obj_mask, object_ids, dtype=tf.bool)
            object_mask = tf.reduce_any(object_mask, axis=0)
            object_masks_i.append(object_mask)
        object_masks.append(tf.stack(object_masks_i, axis=-1))
    pass


def _smoothness(motion_map):
    norm = torch.mean(motion_map.mul(motion_map), dim=[1, 2, 3], keep_dims=True) * 3.0
    motion_map /= torch.sqrt(norm + 1e-12)
    return _smoothness_helper(motion_map)


def _smoothness_helper(motion_map):
    """Calculates L1 (total variation) smoothness loss of a tensor.

  Args:
    motion_map: A tensor to be smoothed, of shape [B, H, W, C].

  Returns:
    A scalar tf.Tensor, The total variation loss.
  """
    # We roll in order to impose continuity across the boundary. The motivation is
    # that there is some ambiguity between rotation and spatial gradients of
    # translation maps. We would like to discourage spatial gradients of the
    # translation field, and to absorb sich gradients into the rotation as much as
    # possible. This is why we impose continuity across the spatial boundary.
    motion_map_dx = motion_map - torch.roll(motion_map, 1, 1)
    motion_map_dy = motion_map - torch.roll(motion_map, 1, 2)
    sm_loss = torch.sqrt(1e-24 + motion_map_dx.mul(motion_map_dx) + motion_map_dy.mul(motion_map_dy))

    return torch.mean(sm_loss)