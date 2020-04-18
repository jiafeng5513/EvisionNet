# -*- coding: utf-8 -*-
"""
Loss Function for EvisionNet

based on "Depth from video in the wild", "SfmLearner-PyTorch", "SfmLearner-TF" and "struct2depth"

Total Loss = Depth Smoothing +          # 2. 深度平滑
             Motion Smoothing +         # 4. 运动平滑(需要seg_stack即object_mask)
             Reconstr loss +            # 1. 重投影损失
             ssim loss +                # 3. structural similarity index,结构相似性损失
             Depth Consistency loss +   # 5.
             Rot_loss +                 # 6. 旋转损失
             Trans_loss                 # 7. 平移损失

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
"""
训练现场:
1. train函数每次处理的是一个epoch,在内部实际上的逐个batch进行反向传播的
2. 对于每个batch,我们需要让depthnet前向传播SEQ_LENGTH次,让motion_net前向传播(SEQ_LENGTH-1)*2次
假设SEQ_LENGTH=5,那么MotionNet需要这样计算: 12/21,23/32,34/43,45/54,共(SEQ_LENGTH-1)*2次
        其中正序的一组存储在rotation_pred,translation_pred,residual_translation_pred,intrinsic_pred中
        逆序的一组存储在inv_rotation_pred,inv_translation_pred,inv_residual_translation_pred,inv_intrinsic_pred中
        这8个list的长度都是SEQ_LENGTH-1
"""

import torch
import torch.nn as nn
from evision_model import transform_depth_map

LAYER_NORM_NOISE_RAMPUP_STEPS = 10000
MIN_OBJECT_AREA = 20


class LossFactory():
    # public : 构造函数
    def __init__(self, SEQ_LENGTH=3, reconstr_weight=0.85, smooth_weight=1e-2, ssim_weight=3.0,
                 motion_smoothing_weight=1e-3, rotation_consistency_weight=1e-3,
                 translation_consistency_weight=1e-2, depth_consistency_loss_weight=1e-2):
        # 构造函数用于传入权重系数
        self.SEQ_LENGTH = SEQ_LENGTH
        self.reconstr_weight = reconstr_weight
        self.smooth_weight = smooth_weight
        self.ssim_weight = ssim_weight
        self.motion_smoothing_weight = motion_smoothing_weight
        self.rotation_consistency_weight = rotation_consistency_weight
        self.translation_consistency_weight = translation_consistency_weight
        self.depth_consistency_loss_weight = depth_consistency_loss_weight,
        pass

    # public : 联合损失
    def getTotalLoss(self):
        # 这里用于计算联合损失函数
        pass

    # private : 深度平滑损失
    def __Depth_Smoothing_Loss(self, depth_pred, image_stack):
        """
        total loss part 1 :深度平滑
        TODO:这里实际上计算的是视差的平滑
        Args:
            image_stack:输入图像,在通道维度上层叠,[b,3*SEQ_LENGTH,h,w]
            depth_pred:深度预测       list of [b,1,h,w], list length = SEQ_LENGTH
        Returns:
            Depth_Smoothing
        """

        smooth_loss = 0
        disp = {}
        for i in range(self.SEQ_LENGTH):
            disp[i] = 1.0 / depth_pred.depth[i]

        for i in range(self.SEQ_LENGTH):
            disp_smoothing = disp[i]
            # Perform depth normalization, dividing by the mean.
            mean_disp = torch.mean(disp_smoothing, axis=[1, 2, 3], keep_dims=True)
            disp_input = disp_smoothing / mean_disp
            smooth_loss += _depth_smoothness(disp_input, image_stack[:, 3 * i:3 * (i + 1), :, :])
        return smooth_loss
        pass

    # private : 场景运动平滑损失
    def __Motion_Smoothing_Loss(self, trans, trans_res, inv_trans, inv_trans_res, seg_stack):
        """
        场景运动平滑损失
        TODO:seg_stack可能存在数组越界
        Args:
            trans:          list of 平移          [b,3,1,1]
            trans_res:      list of 场景运动      [b,3,h,w]
            inv_trans:      list of 反向平移      [b,3,1,1]
            inv_trans_res:  list of 反向场景运动  [b,3,h,w] list length = self.SEQ_LENGTH - 1
            seg_stack:      from __get_object_masks()

        Returns:
            场景运动平滑损失
        """
        motion_smoothing = 0.0
        seg_stack = self.__get_object_masks(seg_stack, self.SEQ_LENGTH)
        for i in range(self.SEQ_LENGTH - 1):
            j = i + 1
            # NOTE : trans, trans_res, inv_trans, inv_trans_res are obtained like this:
            # _, trans, trans_res, _ = motion_field_net(images=tf.concat([image_i, image_j], axis=-1))
            # _, inv_trans, inv_trans_res, _ = motion_field_net(images=tf.concat([image_j, image_i], axis=-1))
            motion_smoothing += _smoothness(trans[i] + trans_res[i] * dilate(seg_stack[:, :, :, j:j + 1]))
            motion_smoothing += _smoothness(inv_trans[i] + inv_trans_res[i] * dilate(seg_stack[:, :, :, i:i + 1]))
        return motion_smoothing

    # private : 重投影损失
    def __Reproject_Loss(self):
        # part N: 重投影损失
        pass

    # private : object mask (seg_stack)
    def __get_object_masks(self, seg_stack, SEQ_LENGTH):
        """
        可运动目标掩码
        Args:
            seg_stack:      from data loader
            SEQ_LENGTH:

        Returns:
        处理好的可运动目标掩码
        """
        batch_size = seg_stack.shape[1]
        object_masks = []
        for i in range(batch_size):
            object_ids = torch.unique(seg_stack[i].reshape([-1]))[0]
            object_masks_i = []
            for j in range(SEQ_LENGTH):
                current_seg = seg_stack[i, j * 3, :, :]  # (H, W)

                def process_obj_mask(obj_id):
                    """Create a mask for obj_id, skipping the background mask."""
                    mask = torch.eq(current_seg, obj_id).mul(~torch.eq(torch.Tensor(0).int(), obj_id)).bool()

                    # Leave out vert small masks, that are most often errors.
                    size = mask.int().sum()
                    masl = mask.mul(torch.gt(size, MIN_OBJECT_AREA)).bool()
                    # if not self.boxify:
                    #     return mask
                    # Complete the mask to its bounding box.
                    binary_obj_masks_y = torch.sum(mask, dim=1, keepdim=True).bool()
                    binary_obj_masks_x = torch.sum(mask, dim=0, keepdim=True).bool()
                    return binary_obj_masks_y.mul(binary_obj_masks_x).bool()

                # object_mask = tf.map_fn(  process_obj_mask, object_ids, dtype=tf.bool)# (N, H, W)
                object_mask = torch.tensor([process_obj_mask(obj_id) for obj_id in object_ids])
                object_mask = torch.sum(object_mask, dim=0).bool()
                object_masks_i.append(object_mask)
            object_masks.append(torch.stack(object_masks_i, dim=-1))
        seg_stack = torch.stack(object_masks, dim=0).float()
        return seg_stack

    pass


# ==============================================================================
#                            Helper Functions
# ==============================================================================


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


def rgbd_consistency_loss(frame1transformed_depth, frame1rgb, frame2depth, frame2rgb):
    """Computes a loss that penalizes RGB and depth inconsistencies betwen frames.

  This function computes 3 losses that penalize inconsistencies between two frames: depth, RGB, and structural similarity.
  It IS NOT SYMMETRIC with respect to both frames.
  In particular, to address occlusions,
  it only penalizes depth and RGB inconsistencies at pixels where frame1 is closer to the camera than frame2.
  (Why? see https://arxiv.org/abs/1904.04998).
  Therefore the intended usage pattern is running it twice - second time with the two frames swapped.

  Args:
    frame1transformed_depth: A transform_depth_map.
        TransformedDepthMap object representing the depth map of frame 1 after it was motion-transformed to frame 2,
        a motion transform that accounts for all camera and object motion that occurred between frame1 and frame2.
        The tensors inside frame1transformed_depth are of shape [B, H, W].
    frame1rgb: A tf.Tensor of shape [B, H, W, C] containing the RGB image at frame1.
    frame2depth: A tf.Tensor of shape [B, H, W] containing the depth map at frame2.
    frame2rgb: A tf.Tensor of shape [B, H, W, C] containing the RGB image at frame2.

  Returns:
    A dicionary from string to tf.Tensor, with the following entries:
      depth_error: A tf scalar, the depth mismatch error between the two frames.
      rgb_error: A tf scalar, the rgb mismatch error between the two frames.
      ssim_error: A tf scalar, the strictural similarity mismatch error between
        the two frames.
      depth_proximity_weight: A tf.Tensor of shape [B, H, W], representing a
        function that peaks (at 1.0) for pixels where there is depth consistency
        between the two frames, and is small otherwise.
      frame1_closer_to_camera: A tf.Tensor of shape [B, H, W, 1], a mask that is
        1.0 when the depth map of frame 1 has smaller depth than frame 2.
  """
    pixel_xy = frame1transformed_depth.pixel_xy
    frame2depth_resampled = _resample_depth(frame2depth, pixel_xy)
    frame2rgb_resampled = contrib_resampler.resampler.resampler(frame2rgb, pixel_xy)

    # f1td.depth is the predicted depth at [pixel_y, pixel_x] for frame2. Now we
    # generate (by interpolation) the actual depth values for frame2's depth, at
    # the same locations, so that we can compare the two depths.

    # We penalize inconsistencies between the two frames' depth maps only if the
    # transformed depth map (of frame 1) falls closer to the camera than the
    # actual depth map (of frame 2). This is intended for avoiding penalizing
    # points that become occluded because of the transform.
    # So what about depth inconsistencies where frame1's depth map is FARTHER from
    # the camera than frame2's? These will be handled when we swap the roles of
    # frame 1 and 2 (more in https://arxiv.org/abs/1904.04998).
    frame1_closer_to_camera = tf.to_float(
        tf.logical_and(
            frame1transformed_depth.mask,
            tf.less(frame1transformed_depth.depth, frame2depth_resampled)))
    depth_error = tf.reduce_mean(
        tf.abs(frame2depth_resampled - frame1transformed_depth.depth) *
        frame1_closer_to_camera)

    rgb_error = (
            tf.abs(frame2rgb_resampled - frame1rgb) * tf.expand_dims(
        frame1_closer_to_camera, -1))
    rgb_error = tf.reduce_mean(rgb_error)

    # We generate a weight function that peaks (at 1.0) for pixels where when the
    # depth difference is less than its standard deviation across the frame, and
    # fall off to zero otherwise. This function is used later for weighing the
    # structural similarity loss term. We only want to demand structural
    # similarity for surfaces that are close to one another in the two frames.
    depth_error_second_moment = _weighted_average(
        tf.square(frame2depth_resampled - frame1transformed_depth.depth),
        frame1_closer_to_camera) + 1e-4
    depth_proximity_weight = (
            depth_error_second_moment /
            (tf.square(frame2depth_resampled - frame1transformed_depth.depth) +
             depth_error_second_moment) * tf.to_float(frame1transformed_depth.mask))

    # If we don't stop the gradient training won't start. The reason is presumably
    # that then the network can push the depths apart instead of seeking RGB
    # consistency.
    depth_proximity_weight = tf.stop_gradient(depth_proximity_weight)

    ssim_error, avg_weight = weighted_ssim(
        frame2rgb_resampled,
        frame1rgb,
        depth_proximity_weight,
        c1=float('inf'),  # These values of c1 and c2 work better than defaults.
        c2=9e-6)
    ssim_error = tf.reduce_mean(ssim_error * avg_weight)

    endpoints = {
        'depth_error': depth_error,
        'rgb_error': rgb_error,
        'ssim_error': ssim_error,
        'depth_proximity_weight': depth_proximity_weight,
        'frame1_closer_to_camera': frame1_closer_to_camera
    }
    return endpoints


def motion_field_consistency_loss(frame1transformed_pixelxy, mask,
                                  rotation1, translation1,
                                  rotation2, translation2):
    """Computes a cycle consistency loss between two motion maps.

  Given two rotation and translation maps (of two frames), and a mapping from
  one frame to the other, this function assists in imposing that the fields at
  frame 1 represent the opposite motion of the ones in frame 2.

  In other words: At any given pixel on frame 1, if we apply the translation and
  rotation designated at that pixel, we land on some pixel in frame 2, and if we
  apply the translation and rotation designated there, we land back at the
  original pixel at frame 1.

  Args:
    frame1transformed_pixelxy: A tf.Tensor of shape [B, H, W, 2] representing
      the motion-transformed location of each pixel in frame 1. It is assumed
      (but not verified) that frame1transformed_pixelxy was obtained by properly
      applying rotation1 and translation1 on the depth map of frame 1.
    mask: A tf.Tensor of shape [B, H, W, 2] expressing the weight of each pixel
      in the calculation of the consistency loss.
    rotation1: A tf.Tensor of shape [B, 3] representing rotation angles.
    translation1: A tf.Tensor of shape [B, H, W, 3] representing translation
      vectors.
    rotation2: A tf.Tensor of shape [B, 3] representing rotation angles.
    translation2: A tf.Tensor of shape [B, H, W, 3] representing translation
      vectors.

  Returns:
    A dicionary from string to tf.Tensor, with the following entries:
      rotation_error: A tf scalar, the rotation consistency error.
      translation_error: A tf scalar, the translation consistency error.
  """

    translation2resampled = contrib_resampler.resampler.resampler(
        translation2, tf.stop_gradient(frame1transformed_pixelxy))
    rotation1field = tf.broadcast_to(
        _expand_dims_twice(rotation1, -2), tf.shape(translation1))
    rotation2field = tf.broadcast_to(
        _expand_dims_twice(rotation2, -2), tf.shape(translation2))
    rotation1matrix = transform_utils.matrix_from_angles(rotation1field)
    rotation2matrix = transform_utils.matrix_from_angles(rotation2field)

    rot_unit, trans_zero = transform_utils.combine(
        rotation2matrix, translation2resampled,
        rotation1matrix, translation1)
    eye = tf.eye(3, batch_shape=tf.shape(rot_unit)[:-2])

    transform_utils.matrix_from_angles(rotation1field)  # Delete this later
    transform_utils.matrix_from_angles(rotation2field)  # Delete this later

    # We normalize the product of rotations by the product of their norms, to make
    # the loss agnostic of their magnitudes, only wanting them to be opposite in
    # directions. Otherwise the loss has a tendency to drive the rotations to
    # zero.
    rot_error = tf.reduce_mean(tf.square(rot_unit - eye), axis=(3, 4))
    rot1_scale = tf.reduce_mean(tf.square(rotation1matrix - eye), axis=(3, 4))
    rot2_scale = tf.reduce_mean(tf.square(rotation2matrix - eye), axis=(3, 4))
    rot_error /= (1e-24 + rot1_scale + rot2_scale)
    rotation_error = tf.reduce_mean(rot_error)

    def norm(x):
        return tf.reduce_sum(tf.square(x), axis=-1)

    # Here again, we normalize by the magnitudes, for the same reason.
    translation_error = tf.reduce_mean(
        mask * norm(trans_zero) /
        (1e-24 + norm(translation1) + norm(translation2)))

    return {
        'rotation_error': rotation_error,
        'translation_error': translation_error
    }


def rgbd_and_motion_consistency_loss(frame1transformed_depth, frame1rgb,
                                     frame2depth, frame2rgb, rotation1,
                                     translation1, rotation2, translation2):
    """A helper that bundles rgbd and motion consistency losses together."""
    endpoints = rgbd_consistency_loss(frame1transformed_depth, frame1rgb,
                                      frame2depth, frame2rgb)
    # We calculate the loss only for when frame1transformed_depth is closer to the
    # camera than frame2 (occlusion-awareness). See explanation in
    # rgbd_consistency_loss above.
    endpoints.update(motion_field_consistency_loss(
        frame1transformed_depth.pixel_xy, endpoints['frame1_closer_to_camera'],
        rotation1, translation1, rotation2, translation2))
    return endpoints


def weighted_ssim(x, y, weight, c1=0.01 ** 2, c2=0.03 ** 2, weight_epsilon=0.01):
    """Computes a weighted structured image similarity measure.

  See https://en.wikipedia.org/wiki/Structural_similarity#Algorithm. The only
  difference here is that not all pixels are weighted equally when calculating
  the moments - they are weighted by a weight function.

  Args:
    x: A tf.Tensor representing a batch of images, of shape [B, H, W, C].
    y: A tf.Tensor representing a batch of images, of shape [B, H, W, C].
    weight: A tf.Tensor of shape [B, H, W], representing the weight of each
      pixel in both images when we come to calculate moments (means and
      correlations).
    c1: A floating point number, regularizes division by zero of the means.
    c2: A floating point number, regularizes division by zero of the second
      moments.
    weight_epsilon: A floating point number, used to regularize division by the
      weight.

  Returns:
    A tuple of two tf.Tensors. First, of shape [B, H-2, W-2, C], is scalar
    similarity loss oer pixel per channel, and the second, of shape
    [B, H-2. W-2, 1], is the average pooled `weight`. It is needed so that we
    know how much to weigh each pixel in the first tensor. For example, if
    `'weight` was very small in some area of the images, the first tensor will
    still assign a loss to these pixels, but we shouldn't take the result too
    seriously.
  """
    if c1 == float('inf') and c2 == float('inf'):
        raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                         'likely unintended.')
    weight = tf.expand_dims(weight, -1)
    average_pooled_weight = _avg_pool3x3(weight)
    weight_plus_epsilon = weight + weight_epsilon
    inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

    def weighted_avg_pool3x3(z):
        wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
        return wighted_avg * inverse_average_pooled_weight

    mu_x = weighted_avg_pool3x3(x)
    mu_y = weighted_avg_pool3x3(y)
    sigma_x = weighted_avg_pool3x3(x ** 2) - mu_x ** 2
    sigma_y = weighted_avg_pool3x3(y ** 2) - mu_y ** 2
    sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
    if c1 == float('inf'):
        ssim_n = (2 * sigma_xy + c2)
        ssim_d = (sigma_x + sigma_y + c2)
    elif c2 == float('inf'):
        ssim_n = 2 * mu_x * mu_y + c1
        ssim_d = mu_x ** 2 + mu_y ** 2 + c1
    else:
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    result = ssim_n / ssim_d
    return tf.clip_by_value((1 - result) / 2, 0, 1), average_pooled_weight


def _avg_pool3x3(x):
    return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')


def _weighted_average(x, w, epsilon=1.0):
    weighted_sum = tf.reduce_sum(x * w, axis=(1, 2), keepdims=True)
    sum_of_weights = tf.reduce_sum(w, axis=(1, 2), keepdims=True)
    return weighted_sum / (sum_of_weights + epsilon)


def _resample_depth(depth, coordinates):
    depth = tf.expand_dims(depth, -1)
    result = contrib_resampler.resampler.resampler(depth, coordinates)
    return tf.squeeze(result, axis=3)


def _expand_dims_twice(x, dim):
    return tf.expand_dims(tf.expand_dims(x, dim), dim)
