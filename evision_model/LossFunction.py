# -*- coding: utf-8 -*-
"""
Loss Function for EvisionNet

based on "Depth from video in the wild", "SfmLearner-PyTorch", "SfmLearner-TF" and "struct2depth"

Total Loss = Depth Smoothing +          # [done] 1. image-aware 深度平滑损失
             Motion Smoothing +         # 2. 背景平移场平滑损失
             Reconstr loss +            # 3. RGB惩罚
             ssim loss +                # 4. 结构相似性平衡惩罚
             Depth Consistency loss +   # 5. 深度一致性损失,基于重投影
             Rot_loss +                 # 6. 旋转损失,基于循环变换
             Trans_loss                 # 7. 平移损失,基于循环变换
            (以上各项均带有权重系数超参数)
code by jiafeng5513

联合无监督学习原理:
数据集中含有视频,我们提取相邻的两帧Ps,Pt来看,
我们有DepthNet,可以根据单帧图像得到深度图:Ds,Dt,有MotionNet能够输入相邻的两帧获得相机的运动r,t,还能获得相机内参K,
inv_K代表K的逆矩阵.
            ┌──────┐         ─────>         ┌──────┐
            │  Ps  │          r,t           │  Pt  │
            └──────┘         <─────         └──────┘
               Ds          inv_r,inv_t         Dt
            Ds * Ps = K * R * Dt * inv_K * Pt +K * t
1. 根据这个式子,我们可以从左边解出Ds,此时这个D's是合成的,我们可以把它跟DepthNet输出的Ds进行比较,这就是深度一致性损失
    实际操作时,我们统计D's[i,j]<Ds[i,j]的位置,只有这些位置的深度值参与深度一致性损失,
2. 为了惩罚D's[i,j]全部大于Ds[i,j]的行为,对于所有D's[i,j]<Ds[i,j]的[i,j],累加|Ps[i,j]-Ps[i,j]|,并在RGB三个通道上做均值,
    这就是rgb_error
3. 进一步的惩罚采用结构相似性.首先计算所有D's[i,j]<Ds[i,j]处深度差值的标准差,并以此计算权重函数,
    该权重函数w[i,j]的特点是当深度差小于标准偏差时,达到1.0，否则很小,这就是ssim loss,
    这一步惩罚的理由是: 如果[i,j]处的重投影深度与预测深度差的太多,这个地方很可能发生了遮挡,
    我们给没发生遮挡的那些位置多加一些损失值,给遮挡的地方少加一点,变相降低了可能发生了遮挡的那些位置在联合损失中的权重.
4. 我们的MotionNet还输出了一个[B,3,h,w]的被称为"平移场"的东西,这个东西代表这么一种含义:
    假设在Ps和Pt这两帧之间,相机完全没有移动,但是物体移动了,那么每个像素移动了多少呢?那就由[b,3,i,j]这个平移向量给出,
    因为表示的是背景的运动,而且是一个由平移向量构成的东西,所以我们有时候也把他叫做"残差平移场",或者"背景平移场",
    那么现在考虑相机的运动,就相当于在每一个[b,3,i,j]平移向量的基础上,再增加一个公共的平移向量(也就是相机平移向量).
    这样的好处是显而易见的,相比于粗放的使用一个[b,3]平移向量描述所有点的平移,我们的这种方法显然更加的细致和合理.
    毕竟由于各种各样的奇怪因素,场景中的像素并不总是满足同一个平移向量.
    这里就引出了另一个问题:当我们对于网络寻找"背景移动"的这种能力不是很自信的时候,需要一种机制确保这个"背景平移场"里面描述的是
    具有"背景移动能力"的那些像素,换句话说,我们先验的认为,在相机不动的时候,并不是所有的像素都能自己运动,如果我们事先找到这个场景中的
    可移动物体的边界,那么就能过滤掉"背景平移场"中那些在这个边界之外的像素的"背景运动"
    (因为这很可能是不合理的,由网络的某种机制引入的错误或者噪音),这个边界就是"object_mask",可以使用实例分割,目标检测等方法获得.
    根据DFV的实验结果,一个简单的方框就够了,根据jiafeng5513的实验,这个东西用处不大.
    总之,需要记住这么一件事:在损失函数计算的时候,只要我们说平移,他的shape就是[B,3,H,W],那就代表某个像素的平移,而不是相机的平移.
5. 损失函数的另一个主要组成部分针对平移和旋转,主要原理基于这样一种事实:
    假如Ps上的一个像素Is经过平移和旋转变换到t视角上的It,我们交换s和t,计算出反变换,It根据这个反变换应该能恰好回到Is的位置
    如果这个过程出现了偏差,就说明我们的平移和旋转不够准确,应该使用损失函数惩罚这种现象.
    这就叫"循环变换一致性损失",分为Rot_loss和Trans_loss
6. 损失函数的最后一个部分是所谓的"平滑损失",对"背景平移场"和深度图施加.
    其原理是基于这样一种先验:我们视野中应该由"一片一片"的联通区域构成,每"一片"区域都可能是来自于同一个物体,
    根据生活经验,我们认为常见的物体应该具有连续平滑的景深,而且这个物体自己运动时应该是整体一起运动的,
    这样的先验就意味着深度图和平移场里面不应该存在大量的离群点,而平滑损失正好能约束这一情况
7. 上边的部分过程对于Ps和Pt是不对称的,因此需要交换二者的角色分别计算一次


"""


import torch
import torch.nn as nn
from evision_model import transform_depth_map
import transform_utils

LAYER_NORM_NOISE_RAMPUP_STEPS = 10000
MIN_OBJECT_AREA = 20


class LossFactory(object):
    # public : 构造函数
    def __init__(self, SEQ_LENGTH=3, reconstr_weight=0.85, smooth_weight=1e-2, ssim_weight=1.5,
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
    def getTotalLoss(self, image_stack, seg_stack, depth_pred,
                     trans, trans_res, rot, inv_trans, inv_trans_res, inv_rot, intrinsic_mat):
        _depth_smoothing_loss = self.__Depth_Smoothing_Loss(depth_pred, image_stack)

        (_motion_smoothing, _depth_consistency_loss, _reconstr_loss,
         _ssim_loss, _rot_loss, _trans_loss) = self.__Motion_Smoothing_Loss(trans, trans_res, rot, inv_trans,
                                                                            inv_trans_res, inv_rot, seg_stack,
                                                                            depth_pred, intrinsic_mat, image_stack)
        _total_loss = \
            max(0.0, self.motion_smoothing_weight) * _motion_smoothing + \
            max(0.0, self.smooth_weight) * _depth_smoothing_loss + \
            max(0.0, self.depth_consistency_loss_weight) * _depth_consistency_loss + \
            max(0.0, self.reconstr_weight) * _reconstr_loss + \
            max(0.0, self.ssim_weight) * _ssim_loss + \
            max(0.0, self.rotation_consistency_weight) * _rot_loss + \
            max(0.0, self.translation_consistency_weight) * _trans_loss
        return _total_loss

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

    # private : 其他损失
    def __Motion_Smoothing_Loss(self, trans, trans_res, rot, inv_trans, inv_trans_res, inv_rot, seg_stack,
                                depth, intrinsic_mat, image_stack):
        """
        场景运动平滑损失
        TODO:seg_stack可能存在数组越界
        Args:
            -----list length = self.SEQ_LENGTH - 1
            trans:          list of 相机平移      [b,3,1,1]
            trans_res:      list of 场景运动      [b,3,h,w]
            rot:            list of 相机旋转      []
            inv_trans:      list of 反向相机平移  [b,3,1,1]
            inv_trans_res:  list of 反向场景运动  [b,3,h,w]
            inv_rot:        list of 反向相机旋转  []
             -----list length = self.SEQ_LENGTH
            depth:          list of 深度         []
            intrinsic_mat: 内参
            image_stack:          输入图片
            seg_stack:      from __get_object_masks()

        Returns:
            场景运动平滑损失
        """
        motion_smoothing = 0.0
        depth_consistency_loss = 0.0
        reconstr_loss = 0.0
        ssim_loss = 0.0
        rot_loss = 0.0
        trans_loss = 0.0
        seg_stack = self.__get_object_masks(seg_stack, self.SEQ_LENGTH)
        for i in range(self.SEQ_LENGTH - 1):
            j = i + 1
            # NOTE : trans, trans_res, inv_trans, inv_trans_res are obtained like this:
            # _, trans, trans_res, _ = motion_field_net(images=tf.concat([image_i, image_j], axis=-1))
            # _, inv_trans, inv_trans_res, _ = motion_field_net(images=tf.concat([image_j, image_i], axis=-1))
            _trans = trans[i] + trans_res[i] * dilate(seg_stack[:, :, :, j:j + 1])
            motion_smoothing += _smoothness(_trans)
            _inv_trans = inv_trans[i] + inv_trans_res[i] * dilate(seg_stack[:, :, :, i:i + 1])
            motion_smoothing += _smoothness(_inv_trans)

            image_i = image_stack[:, i * 3:(i + 1) * 3, :, :]
            image_j = image_stack[:, 3 * j:3 * (j + 1), :, :]

            """    
            if SEQ_LENGTH == 3:        
            ┌─────┐         ─────>         ┌─────┐         ─────>         ┌─────┐
            │  i  │  trans,trans_res,rot   │  j  │  trans,trans_res,rot   │     │
            └─────┘         <─────         └─────┘         <─────         └─────┘
            depth[i]        inv_s          depth[j]         inv_s          
            """
            transformed_depth_j = transform_depth_map.using_motion_vector(depth[j], _trans, rot[i], intrinsic_mat)
            endpoints_j = rgbd_and_motion_consistency_loss(
                transformed_depth_j, image_j, depth[i], image_i, rot[i], trans[i], inv_rot[i], inv_trans[i])

            transformed_depth_i = transform_depth_map.using_motion_vector(depth[i], _inv_trans, inv_rot[i],
                                                                          intrinsic_mat)
            endpoints_i = rgbd_and_motion_consistency_loss(
                transformed_depth_i, image_i, depth[j], image_j, inv_rot[i], inv_trans[i], rot[i], trans[i])

            depth_consistency_loss += (endpoints_j['depth_error'] + endpoints_i['depth_error'])
            reconstr_loss += (endpoints_j['rgb_error'] + endpoints_i['rgb_error'])
            ssim_loss += (endpoints_j['ssim_error'] + endpoints_i['ssim_error'])
            rot_loss += (endpoints_j['rotation_error'] + endpoints_i['rotation_error'])
            trans_loss += (endpoints_j['translation_error'] + endpoints_i['translation_error'])

        return motion_smoothing, depth_consistency_loss, reconstr_loss, ssim_loss, rot_loss, trans_loss

    # private : object mask (seg_stack)
    def __get_object_masks(self, seg_stack):
        """
        可运动目标掩码
        Args:
            seg_stack:      from data loader
        Returns:
        处理好的可运动目标掩码
        """
        batch_size = seg_stack.shape[1]
        object_masks = []
        for i in range(batch_size):
            object_ids = torch.unique(seg_stack[i].reshape([-1]))[0]
            object_masks_i = []
            for j in range(self.SEQ_LENGTH):
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
    """
    计算损失，惩罚帧间的RGB和深度不一致。
    此函数计算3种损失，惩罚两个帧之间的不一致：深度，RGB和结构相似性。
    关于这两个帧，它不是对称的。
    特别是要解决遮挡问题，它只会惩罚frame1比frame2更靠近相机的像素处的深度和RGB不一致。
    因此，正确的使用模式是调用两次并交换两个帧的顺序。

    Args:
        frame1transformed_depth: 一个TransformedDepthMap 对象,内部数据相当于[B，H，W],表示将frame 1变换到frame 2之后的深度图，
                                该变换考虑了发生在frame 1和frame 2之间的所有相机和对象运动。 。
        frame1rgb:   张量[B, H, W, C]  RGB image at frame1.
        frame2rgb:   张量[B, H, W, C]  RGB image at frame2.
        frame2depth: 张量[B, H, W]     depth map at frame2.


    Returns:
       张量dict:
          depth_error: A tf scalar, the depth mismatch error between the two frames.
          rgb_error: A tf scalar, the rgb mismatch error between the two frames.
          ssim_error: A tf scalar, the strictural similarity mismatch error between the two frames.
          depth_proximity_weight: 一个张量 [B, H, W], 两帧之间需要计算深度一致性的位置为1.0,其他位置较小
          frame1_closer_to_camera: 一个张量 [B, H, W, 1], frame 1 的深度值比frame 2小的位置为1
    """
    pixel_xy = frame1transformed_depth.pixel_xy
    frame2depth_resampled = _resample_depth(frame2depth, pixel_xy)
    frame2rgb_resampled = contrib_resampler.resampler.resampler(frame2rgb, pixel_xy)

    # f1td.depth 是frame2的预测深度at [pixel_y, pixel_x].
    # 现在我们通过插值生成同样坐标网格下frame2的深度,以便进行比较

    # We penalize inconsistencies between the two frames' depth maps only if the
    # transformed depth map (of frame 1) falls closer to the camera than the
    # actual depth map (of frame 2). This is intended for avoiding penalizing
    # points that become occluded because of the transform.
    # So what about depth inconsistencies where frame1's depth map is FARTHER from
    # the camera than frame2's? These will be handled when we swap the roles of
    # frame 1 and 2 (more in https://arxiv.org/abs/1904.04998).
    frame1_closer_to_camera = tf.to_float(
        tf.logical_and(frame1transformed_depth.mask,
                        tf.less(frame1transformed_depth.depth, frame2depth_resampled)))
    depth_error = tf.reduce_mean(tf.abs(frame2depth_resampled - frame1transformed_depth.depth) * frame1_closer_to_camera)
    rgb_error = (tf.abs(frame2rgb_resampled - frame1rgb) * tf.expand_dims(frame1_closer_to_camera, -1))
    rgb_error = tf.reduce_mean(rgb_error)

    # We generate a weight function that peaks (at 1.0) for pixels where when the
    # depth difference is less than its standard deviation across the frame, and
    # fall off to zero otherwise. This function is used later for weighing the
    # structural similarity loss term. We only want to demand structural
    # similarity for surfaces that are close to one another in the two frames.
    depth_error_second_moment = _weighted_average(tf.square(frame2depth_resampled - frame1transformed_depth.depth),
                                                  frame1_closer_to_camera) \
                                + 1e-4
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
    """
        计算两个运动图之间的循环一致性损失。


    在第1帧的任何给定像素处，如果我们应用在该像素处指定的平移和旋转，
    则变换到第2帧的某个像素处，然后在该像素上应用在该像素处指定的平移和旋转，我们将回到第1帧的原始像素那个位置.

  Args:
    frame1transformed_pixelxy: A tf.Tensor of shape [B, H, W, 2]
    representing the motion-transformed location of each pixel in frame 1.
    It is assumed (but not verified) that
    frame1transformed_pixelxy was obtained by properlyapplying rotation1 and translation1 on the depth map of frame 1.
    mask: 张量 [b，H，W，2] 表示一致性损失的计算中的每个像素的权重。
    rotation1:  [B, 3] 旋转角 1->2
    translation1: [B, H, W, 3] 平移向量场(由每个像素的平移向量构成),1->2
    rotation2: [B, 3] 旋转角 2->1
    translation2: [B, H, W, 3] 平移向量场(由每个像素的平移向量构成),2->1

  Returns:
    A dicionary from string to tf.Tensor, with the following entries:
      rotation_error: A tf scalar, the rotation consistency error.
      translation_error: A tf scalar, the translation consistency error.
  """

    translation2resampled = \
        contrib_resampler.resampler.resampler(translation2, tf.stop_gradient(frame1transformed_pixelxy))
    rotation1field = tf.broadcast_to(_expand_dims_twice(rotation1, -2), translation1.shape)
    rotation2field = tf.broadcast_to(_expand_dims_twice(rotation2, -2), translation2.shape)
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
    endpoints = rgbd_consistency_loss(frame1transformed_depth, frame1rgb, frame2depth, frame2rgb)
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
        raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is likely unintended.')
    weight = torch.unsqueeze(weight, 1)
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
    return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight


def _avg_pool3x3(x):
    # tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')
    return torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=0)


def _weighted_average(x, w, epsilon=1.0):
    # weighted_sum = tf.reduce_sum(x * w, axis=(1, 2), keepdims=True)
    weighted_sum = (x * w).sum(dim=(1, 2), keepdims=True)
    # sum_of_weights = tf.reduce_sum(w, axis=(1, 2), keepdims=True)
    sum_of_weights = w.sum(dim=(1, 2), keepdims=True)
    return weighted_sum / (sum_of_weights + epsilon)


def _resample_depth(depth, coordinates):
    depth = torch.unsqueeze(depth, 1)
    result = contrib_resampler.resampler.resampler(depth, coordinates)
    return torch.squeeze(result, dim=1)


def _expand_dims_twice(x, dim):
    return torch.unsqueeze(torch.unsqueeze(x, dim), dim)
