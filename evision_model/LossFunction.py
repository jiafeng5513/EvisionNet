# -*- coding: utf-8 -*-
"""
Loss Function for EvisionNet

based on "Depth from video in the wild", "SfmLearner-PyTorch", "SfmLearner-TF" and "struct2depth"

Total Loss = Reprojection error +

code by jiafeng5513

NOTE:
    1. TensorFlow 的默认顺序是 [N H W C], PyTorch的默认顺序是 [N C H W]
    2. 计算损失的最小单位是一个SEQ_LENGTH, 预处理时生成的一张连体照片由SEQ_LENGTH个连续帧组成
    3. image的联结方式是在C维度,也就是通道上进行连接
    4. 由于2,每次计算损失时,关于深度的损失有SEQ_LENGTH个,关于ego-motion的损失有SEQ_LENGTH-1个
"""
import torch
import torch.nn as nn


def getTotalLoss(SEQ_LENGTH,  # 单位训练序列长度,目前默认为3,需要支持其他数值
                 depth_pred,  # 深度预测
                 smooth_weight,  # 平滑损失权重
                 image_stack,  # 训练图片0-1-2
                 seg_stack,  # ???
                 rot, trans, trans_res, mat,  # 0-1运动预测
                 inv_rot, inv_trans, inv_trans_res, inv_mat,  # 1-2运动预测
                 GT_intrinsic_mat,  # GT内参
                 learn_intrinsics=True,  # 是否学习内参预测
                 foreground_dilation=8,
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
    # for i in range(SEQ_LENGTH - 1):
    #     j = i + 1
    #     depth_i = depth_pred[i][:, 0, :, :]
    #     depth_j = depth_pred[j][:, 0, :, :]
    #     image_j = image_stack[:, 3 * j:3 * (j + 1), :, :]
    #     image_i = image_stack[:, i * 3:(i + 1) * 3, :, :]
    #     We select a pair of consecutive images (and their respective predicted depth maps).
    #     Now we have the network predict a motion field that connects the two.
    # We feed the pair of images into the network, once in forward order and then in reverse order.
    # The results are fed into the loss calculation.
    # The following losses are calculated:
    # - RGB and SSIM photometric consistency.
    # - Cycle consistency of rotations and translations for every pixel.
    # - L1 smoothness of the disparity and the motion field.
    # - Depth consistency
    # rot, trans, trans_res, mat = motion_prediction_net.motion_field_net(
    #     images=tf.concat([image_i, image_j], axis=-1),#注意这种写法,非常的灵性,包括上边的那段
    #     weight_reg=self.weight_reg)
    # inv_rot, inv_trans, inv_trans_res, inv_mat = (
    #     motion_prediction_net.motion_field_net(
    #         images=tf.concat([image_j, image_i], axis=-1),
    #         weight_reg=self.weight_reg))
    if learn_intrinsics:
        intrinsic_mat = 0.5 * (mat + inv_mat)
    else:
        intrinsic_mat = GT_intrinsic_mat[:, 0, :, :]

    # 我们需要知道这是用来干什么的,如有替代的可能,则代替之,否则,kitti数据无法使用

    trans += trans_res * dilate(self.seg_stack[:, :, :, j:j + 1], foreground_dilation)

    inv_trans += inv_trans_res * dilate(self.seg_stack[:, :, :, i:i + 1], foreground_dilation)
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