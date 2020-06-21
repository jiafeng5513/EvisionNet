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
数据集中含有视频,我们提取相邻的两帧Is,It来看,
我们有DepthNet,可以根据单帧图像得到深度图:Ds,Dt,有MotionNet能够输入相邻的两帧获得相机的运动r,t,还能获得相机内参K,
inv_K代表K的逆矩阵.Ps和Pt代表Is和It上的点的坐标.
            ┌──────┐         ─────>         ┌──────┐
            │  I1  │          r,t           │  I2  │
            └──────┘         <─────         └──────┘
               D1          inv_r,inv_t         D2
            D1 * P1 = K * R * D2 * inv_K * P2 +K * 2
正确认识这个式子的含义是理解EvisionNet训练理论的基础.
    问1: 这个公式的作用对象是什么?
    答1: 已知K R t的情况下,机位2拍摄的图片中一点P2具有深度值D2,我们可以根据这个公式,算出同一点在机位1的位置上所射图片中的像素坐标,
         并获得这一点在机位1处对应的深度值.

    问2: 所以这个过程和RGB值无关是吗?
    答2: 是的,这个过程不涉及RGB像素值的问题,这个过程也并不能直接进行两个RGB图像之间的映射.

    问3: 等式右边的P2坐标一定是整数的,等式右边的坐标P1可能是浮点数的,怎么办?
    答3: 可以通过插值,把P1落回到整数网格坐标,具体来讲:
        a.如果我们对落在浮点坐标上的深度值插值,将得到一张能够与原始深度形成对比关系的"重投影深度图"
        b.根据已经获得的像素坐标对应关系,我们能够把RGB图像重投影
        c.和深度图一样,上述"重投影RGB图像"也落在了浮点坐标上,如果要与真实的RGB图像进行对比,也要进行插值采样
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
    def __init__(self, SEQ_LENGTH=3, rgb_weight=0.85, depth_smoothing_weight=1e-2, ssim_weight=1.5,
                 motion_smoothing_weight=1e-3, rotation_consistency_weight=1e-3,
                 translation_consistency_weight=1e-2, depth_consistency_loss_weight=1e-2):
        # 构造函数用于传入权重系数
        self.SEQ_LENGTH = SEQ_LENGTH
        self.rgb_weight = rgb_weight
        self.depth_smoothing_weight = depth_smoothing_weight
        self.ssim_weight = ssim_weight
        self.motion_smoothing_weight = motion_smoothing_weight
        self.rotation_consistency_weight = rotation_consistency_weight
        self.translation_consistency_weight = translation_consistency_weight
        self.depth_consistency_loss_weight = depth_consistency_loss_weight

        self.MotionSmoothingLoss = 0.0
        self.DepthSmoothingLoss = 0.0
        self.DepthConsistencyLoss = 0.0
        self.RgbPenalty = 0.0
        self.SsimPenalty = 0.0
        self.RotLoss = 0.0
        self.TransLoss = 0.0
        self.TotalLoss = 0.0
        pass

    # geters
    def getMotionSmoothingLoss(self):
        return self.MotionSmoothingLoss

    def getDepthSmoothingLoss(self):
        return self.DepthSmoothingLoss

    def getDepthConsistencyLoss(self):
        return self.DepthConsistencyLoss

    def getRgbPenalty(self):
        return self.RgbPenalty

    def getSsimPenalty(self):
        return self.SsimPenalty

    def getRotLoss(self):
        return self.RotLoss

    def getTransLoss(self):
        return self.TransLoss

    def getTotalLoss(self):
        return self.TotalLoss

    # public : 联合损失
    def SolveTotalLoss(self, image_stack, depth_pred,
                     trans, trans_res, rot, inv_trans, inv_trans_res, inv_rot,
                     intrinsic_mat, seg_stack=None):
        """
        计算联合损失函数,联合损失函数由七个部分组成,详情请看本文件开头的注释
            ┌─────┐         ─────>         ┌─────┐         ─────>         ┌─────┐
            │  i  │  trans,trans_res,rot   │  j  │  trans,trans_res,rot   │     │
            └─────┘         <─────         └─────┘         <─────         └─────┘
            depth[i]        inv_s          depth[j]         inv_s
        Args:
            image_stack:    输入图片,         list of [B,3,h,w], length = SEQ_LENGTH
            depth_pred:     DepthNet深度预测, list of [B,1,h,w], length = SEQ_LENGTH
            trans:          i->j相机平移向量, list of [B,3],     length = SEQ_LENGTH - 1
            trans_res:      i->j背景平移场,   list of [B,1,h,w], length = SEQ_LENGTH - 1
            rot:            i->j相机旋转矩阵, list of [b,3,3],   length = SEQ_LENGTH - 1
            inv_trans:      j->i相机平移向量, list of [B,3],     length = SEQ_LENGTH - 1
            inv_trans_res:  j->i背景平移场,   list of [B,1,h,w], length = SEQ_LENGTH - 1
            inv_rot:        j->i相机旋转矩阵, list of [b,3,3],   length = SEQ_LENGTH - 1
            intrinsic_mat: 相机内参矩阵,如果使用网络预测结果,则为SEQ_LENGTH上的平均值,否则为GroundTruth
            seg_stack:      分割掩码,0 for immovable,1 for movable,list of [B,1,h,w], length = SEQ_LENGTH
        Returns:
            联合损失函数值
        """
        _depth_smoothing_loss = 0.0
        for i in range(self.SEQ_LENGTH):
            _depth_smoothing_loss += self.__Depth_Smoothing_Loss(depth_pred[i], image_stack[i])
            pass

        _motion_smoothing = 0.0
        _depth_consistency_loss = 0.0
        _rgb_loss = 0.0
        _ssim_loss = 0.0
        _rot_loss = 0.0
        _trans_loss = 0.0
        for i in range(self.SEQ_LENGTH - 1):
            # ----------------_motion_smoothing--------------------#
            if seg_stack is not None:
                _trans = trans[i] + trans_res[i] * dilate(seg_stack[i + 1])
                _inv_trans = inv_trans[i] + inv_trans_res[i] * dilate(seg_stack[i])
            else:
                _trans = trans[i] + trans_res[i]
                _inv_trans = inv_trans[i] + inv_trans_res[i]
            _motion_smoothing += self.__Motion_field_smoothness(_trans)
            _motion_smoothing += self.__Motion_field_smoothness(_inv_trans)
            # ---------------------[i-->j]-------------------------#
            transformed_depth_j = transform_depth_map.using_motion_vector(depth_pred[i + 1], _trans, rot[i],
                                                                          intrinsic_mat)

            _depth_consistency_loss_j, _rgb_loss_j, _ssim_loss_j, mask_j = \
                self.__Depth_Consistency_Loss(transformed_depth_j, image_stack[i + 1], depth_pred[i], image_stack[i])

            _rot_loss_j, _trans_loss_j = self.__Cyclic_Consistency_Loss(transformed_depth_j.pixel_xy, mask_j, rot[i],
                                                                        _trans, inv_rot[i], _inv_trans)
            # ---------------------[j-->i]-------------------------#
            transformed_depth_i = transform_depth_map.using_motion_vector(depth_pred[i], _inv_trans, inv_rot[i],
                                                                          intrinsic_mat)

            _depth_consistency_loss_i, _rgb_loss_i, _ssim_loss_i, mask_i = \
                self.__Depth_Consistency_Loss(transformed_depth_j, image_stack[i], depth_pred[i + 1],
                                              image_stack[i + 1])

            _rot_loss_i, _trans_loss_i = self.__Cyclic_Consistency_Loss(transformed_depth_i.pixel_xy, mask_i,
                                                                        inv_rot[i],
                                                                        _inv_trans, rot[i], _trans)
            # -----------------------累加--------------------------#
            _depth_consistency_loss += (_depth_consistency_loss_j + _depth_consistency_loss_i)
            _rgb_loss += (_rgb_loss_j + _rgb_loss_i)
            _ssim_loss += (_ssim_loss_j + _ssim_loss_i)
            _rot_loss += (_rot_loss_j + _rot_loss_i)
            _trans_loss += (_trans_loss_j + _trans_loss_i)

        # 加权
        self.MotionSmoothingLoss = max(0.0, self.motion_smoothing_weight) * _motion_smoothing
        self.DepthSmoothingLoss = max(0.0, self.depth_smoothing_weight) * _depth_smoothing_loss
        self.DepthConsistencyLoss = max(0.0, self.depth_consistency_loss_weight) * _depth_consistency_loss
        self.RgbPenalty = max(0.0, self.rgb_weight) * _rgb_loss
        self.SsimPenalty = max(0.0, self.ssim_weight) * _ssim_loss
        self.RotLoss = max(0.0, self.rotation_consistency_weight) * _rot_loss
        self.TransLoss = max(0.0, self.translation_consistency_weight) * _trans_loss

        self.TotalLoss =  self.MotionSmoothingLoss + self.DepthSmoothingLoss + self.DepthConsistencyLoss + \
                      self.RgbPenalty + self.SsimPenalty + self.RotLoss + self.TransLoss
        pass

    # private : 1.深度平滑损失
    def __Depth_Smoothing_Loss(self, depth, image):
        """
        total loss part 1 :深度平滑
        TODO:这里实际上计算的是视差的平滑
        Args:
            image:       输入图像      [b,3,h,w]
            depth:       深度预测      [b,1,h,w]
        Returns:
            Depth_Smoothing
        """
        disp = 1.0 / depth
        # Perform depth normalization, dividing by the mean.
        mean_disp = torch.mean(disp, dim=(1, 2, 3), keepdim=True)
        disp_input = disp / mean_disp
        # Computes image-aware depth smoothness loss.
        depth_dx = disp_input[:, :, :, :-1] - disp_input[:, :, :, 1:]
        depth_dy = disp_input[:, :, :-1, :] - disp_input[:, :, 1:, :]
        image_dx = image[:, :, :, :-1] - image[:, :, :, 1:]
        image_dy = image[:, :, :-1, :] - image[:, :, 1:, :]
        weights_x = torch.exp(-torch.mean(torch.abs(image_dx), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_dy), 1, keepdim=True))
        smoothness_x = depth_dx * weights_x
        smoothness_y = depth_dy * weights_y
        smooth_loss = torch.mean(abs(smoothness_x)) + torch.mean(abs(smoothness_y))
        return smooth_loss

    # private : 2.背景平移场平滑损失
    def __Motion_field_smoothness(self, motion_map):
        norm = torch.mean(motion_map.pow(2), dim=(1, 2, 3), keepdim=True) * 3.0
        motion_map_p = motion_map / torch.sqrt(norm + 1e-12)

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
        motion_map_dx = motion_map_p - torch.roll(motion_map_p, 1, 1)
        motion_map_dy = motion_map_p - torch.roll(motion_map_p, 1, 2)
        sm_loss = torch.sqrt(1e-24 + motion_map_dx.pow(2) + motion_map_dy.pow(2))

        return torch.mean(sm_loss)

    # private : 3.4.5.深度一致性损失(基于重投影),RGB惩罚,SSIM
    def __Depth_Consistency_Loss(self, frame1transformed_depth, frame1rgb, frame2depth, frame2rgb):
        """
            计算损失，惩罚帧间的RGB和深度不一致。
            此函数计算3种损失，惩罚两个帧之间的不一致：深度，RGB和结构相似性。
            关于这两个帧，它不是对称的。
            特别是要解决遮挡问题，它只会惩罚frame1比frame2更靠近相机的像素处的深度和RGB不一致。
            因此，正确的使用模式是调用两次并交换两个帧的顺序。
            Args:
                frame1transformed_depth: 一个TransformedDepthMap 对象,内部数据相当于[B，H，W],表示将frame 1变换到frame 2之后的深度图，
                                        该变换考虑了发生在frame 1和frame 2之间的所有相机和对象运动。 。
                frame1rgb:   张量[B, C, H, W]  RGB image at frame1.
                frame2rgb:   张量[B, C, H, W]  RGB image at frame2.
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
        frame2depth_resampled = torch.squeeze(resample(frame2depth, pixel_xy), dim=1)
        frame2rgb_resampled = resample(frame2rgb, pixel_xy)

        frame1_closer_to_camera = frame1transformed_depth.mask.mul(torch.le(frame1transformed_depth.depth,
                                                                            frame2depth_resampled)).float()
        depth_error = (
                torch.abs(frame2depth_resampled - frame1transformed_depth.depth) * frame1_closer_to_camera).mean()
        rgb_error = (torch.abs(frame2rgb_resampled - frame1rgb) * torch.unsqueeze(frame1_closer_to_camera, 1))
        rgb_error = rgb_error.mean()

        def _weighted_average(x, w, epsilon=1.0):
            # weighted_sum = tf.reduce_sum(x * w, axis=(1, 2), keepdims=True)
            weighted_sum = (x * w).sum(dim=(1, 2), keepdims=True)
            # sum_of_weights = tf.reduce_sum(w, axis=(1, 2), keepdims=True)
            sum_of_weights = w.sum(dim=(1, 2), keepdims=True)
            return weighted_sum / (sum_of_weights + epsilon)

        depth_error_second_moment = _weighted_average((frame2depth_resampled - frame1transformed_depth.depth).pow(2),
                                                      frame1_closer_to_camera) + 1e-4

        depth_proximity_weight = (
                depth_error_second_moment / ((frame2depth_resampled - frame1transformed_depth.depth).pow(2) +
                                             depth_error_second_moment) * frame1transformed_depth.mask.float())
        ssim_error, avg_weight = self.__weighted_ssim(frame2rgb_resampled,
                                                      frame1rgb,
                                                      depth_proximity_weight,  # stop_gradient
                                                      c1=float('inf'),  # this c1 and c2 work better than defaults.
                                                      c2=9e-6)
        ssim_error = (ssim_error * avg_weight).mean()
        return depth_error, rgb_error, ssim_error, frame1_closer_to_camera

    # private : 6.7.循环变换一致性损失
    def __Cyclic_Consistency_Loss(self, frame1transformed_depth_pixelxy, mask, rotation1, translation1, rotation2,
                                  translation2):
        """
               计算两个运动图之间的循环一致性损失。
           在第1帧的任何给定像素处，如果我们应用在该像素处指定的平移和旋转，
           则变换到第2帧的某个像素处，然后在该像素上应用在该像素处指定的平移和旋转，我们将回到第1帧的原始像素那个位置.

         Args:
           frame1transformed_depth_pixelxy: A tf.Tensor of shape [B, H, W, 2]
           representing the motion-transformed location of each pixel in frame 1.
           It is assumed (but not verified) that
           frame1transformed_pixelxy was obtained by properlyapplying rotation1 and translation1 on the depth map of frame 1.
           mask: 张量 [b，H，W，2] 表示一致性损失的计算中的每个像素的权重。__Depth_Consistency_Loss里面的frame1_closer_to_camera
           rotation1:  [B, 3] 旋转角 1->2
           translation1: [B, H, W, 3] 平移向量场(由每个像素的平移向量构成),1->2
           rotation2: [B, 3] 旋转角 2->1
           translation2: [B, H, W, 3] 平移向量场(由每个像素的平移向量构成),2->1

         Returns:
           A dicionary from string to tf.Tensor, with the following entries:
             rotation_error: A tf scalar, the rotation consistency error.
             translation_error: A tf scalar, the translation consistency error.
         """
        translation2resampled = resample(translation2.permute(0, 3, 1, 2),
                                         frame1transformed_depth_pixelxy).permute(0, 2, 3, 1)  # stop_gradient

        def _expand_dims_twice(x, dim):
            return torch.unsqueeze(torch.unsqueeze(x, dim), dim)

        rotation1field, _ = torch.broadcast_tensors(_expand_dims_twice(rotation1, -2),
                                                    translation1)  # translation1 [4,128,416,3]
        rotation2field, _ = torch.broadcast_tensors(_expand_dims_twice(rotation2, -2), translation2)
        rotation1matrix = transform_utils.matrix_from_angles(rotation1field)
        rotation2matrix = transform_utils.matrix_from_angles(rotation2field)

        rot_unit, trans_zero = transform_utils.combine(rotation2matrix, translation2resampled,
                                                       rotation1matrix, translation1)

        eye_shape = rot_unit.shape[:-2] + (1, 1)
        eye = torch.eye(3).unsqueeze(0).repeat(eye_shape).cuda()  # TODO:自适应运算设备

        transform_utils.matrix_from_angles(rotation1field)  # Delete this later
        transform_utils.matrix_from_angles(rotation2field)  # Delete this later

        # We normalize the product of rotations by the product of their norms, to make
        # the loss agnostic of their magnitudes, only wanting them to be opposite in
        # directions. Otherwise the loss has a tendency to drive the rotations to
        # zero.
        rot_error = torch.mean((rot_unit - eye).pow(2), dim=(3, 4))
        rot1_scale = torch.mean((rotation1matrix - eye).pow(2), dim=(3, 4))
        rot2_scale = torch.mean((rotation2matrix - eye).pow(2), dim=(3, 4))
        rot_error /= (1e-24 + rot1_scale + rot2_scale)
        rotation_error = rot_error.mean()

        def norm(x):
            return x.pow(2).sum(-1)

        # Here again, we normalize by the magnitudes, for the same reason.
        translation_error = (mask * norm(trans_zero) / (1e-24 + norm(translation1) + norm(translation2))).mean()

        return rotation_error, translation_error

    # private
    def __weighted_ssim(self, x, y, weight, c1=0.01 ** 2, c2=0.03 ** 2, weight_epsilon=0.01):
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
        average_pooled_weight = torch.nn.functional.avg_pool2d(weight, kernel_size=3, stride=1, padding=0)
        weight_plus_epsilon = weight + weight_epsilon
        inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

        def weighted_avg_pool3x3(z):
            wighted_avg = torch.nn.functional.avg_pool2d(z * weight_plus_epsilon, kernel_size=3, stride=1, padding=0)
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

    pass


# ==============================================================================
#                            Helper Functions
# ==============================================================================

def dilate(x, foreground_dilation):
    # Dilation by n pixels is roughtly max pooling by 2 * n + 1.
    p = foreground_dilation * 2 + 1
    pool2d = nn.MaxPool2d((p, p), stride=(1, 1))  # TODO:tf.nn.max_pool(x, [1, p, p, 1], [1] * 4, 'SAME')
    return pool2d(x)


def resample(int_map=None, xy_float=None):
    """
        int_map上的像素处于整数网格坐标下,xy_float描述了一个浮点数网格
        该函数的作用就是把int_map在xy_float下进行重新采样
    Args:
        int_map      :   输入张量, [B,C,H_in,W_in],C可能是1或者3
        xy_float     :   浮点数网格坐标,[B,H_out,W_out,2],需要注意,这是坐标网格,不遵守[b.c.h.w]
    Returns:
        调整后的数据  :   [B,C,H_out,W_out]
    """
    b_1, c_1, H_in, W_in = int_map.shape
    b_2, H_out, W_out, c_2 = xy_float.shape

    # 把坐标归一化到[-1,1],数据类型保持在float
    x_float, y_float = torch.unbind(xy_float, dim=3)  # 把x和y 拆看,分别广播操作
    x_float = torch.unsqueeze((2 * x_float / (W_in - 1)) - 1, dim=-1)  # new_x = 2*x/(w-1)-1
    y_float = torch.unsqueeze((2 * y_float / (H_in - 1)) - 1, dim=-1)  # new_y = 2*y/(h-1)-1
    grid = torch.cat((x_float, y_float), dim=-1).float()
    output = torch.nn.functional.grid_sample(int_map.float(), grid,
                                             mode='bilinear', padding_mode='zeros', align_corners=False)
    # align_corners=False是默认行为,True是PyTorch 1.3.0之前的默认行为,然而该函数的行为通过测试的版本是1.5,因此显示指定
    return output
