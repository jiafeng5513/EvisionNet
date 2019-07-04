from __future__ import division
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
import sys
import requests
from depth_evaluation_utils import *
from glob import glob
from pose_evaluation_utils import *


def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='gray'):
    # convert to disparity
    depth = 1. / (depth + 1e-6)
    if normalizer is not None:
        depth = depth / normalizer
    else:
        depth = depth / (np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1 - crop_percent))
    depth = depth[:keep_H]
    depth = depth
    return depth


def euler2mat(z, y, x):
    """Converts euler angles to rotation matrix
   TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
  Args:
      z: rotation angle along z axis (in radians) -- size = [B, N]
      y: rotation angle along y axis (in radians) -- size = [B, N]
      x: rotation angle along x axis (in radians) -- size = [B, N]
  Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
  """
    B = tf.shape(z)[0]
    N = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([B, N, 1, 1])
    ones = tf.ones([B, N, 1, 1])

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz, cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny, zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
    return rotMat


def pose_vec2mat(vec):
    """
  Converts 6DoF parameters to transformation matrix
  B for Batch Size
  Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
  Returns:
      A transformation matrix -- [B, 4, 4]
  """
    batch_size, _ = vec.get_shape().as_list()
    translation = tf.slice(vec, [0, 0], [-1, 3])  # 切出只含有平移的部分,尺寸为[B,3]
    translation = tf.expand_dims(translation, -1) # 最后加一个维度,[B,4]
    rx = tf.slice(vec, [0, 3], [-1, 1])  # 从vec里面把rx切下来,[B,1],下同
    ry = tf.slice(vec, [0, 4], [-1, 1])
    rz = tf.slice(vec, [0, 5], [-1, 1])
    rot_mat = euler2mat(rz, ry, rx)     # 欧拉角转旋转矩阵,z,y,x都是[B,1],一组x,y,z转成一个3x3的旋转矩阵,结果是[B,1,3,3]
    rot_mat = tf.squeeze(rot_mat, axis=[1])  #删除1位的那个维度,变成[B,3,3]
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])  # 创建[1,1,4]的变量
    filler = tf.tile(filler, [batch_size, 1, 1])  # 创建[4,1,4]的变量[[[0. 0. 0. 1.]]
                                                  #                  [[0. 0. 0. 1.]]
                                                  #                  [[0. 0. 0. 1.]]
                                                  #                  [[0. 0. 0. 1.]]]
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)  # [B,4,4],最后一行是0,0,0,1,一共4个有效数字
    return transform_mat #[]


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.

  Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """
    batch, height, width = depth.get_shape().as_list()
    depth = tf.reshape(depth, [batch, 1, -1])
    pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
    cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
    if is_homogeneous:
        ones = tf.ones([batch, 1, height * width])
        cam_coords = tf.concat([cam_coords, ones], axis=1)
    cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
    return cam_coords


def cam2pixel(cam_coords, proj):
    """Transforms coordinates in a camera frame to the pixel frame.

  Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
  Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
  """
    batch, _, height, width = cam_coords.get_shape().as_list()
    cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
    unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
    x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
    y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
    z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    pixel_coords = tf.concat([x_n, y_n], axis=1)
    pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
    return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])


def meshgrid(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid.

  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                    tf.transpose(tf.expand_dims(
                        tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                    tf.ones(shape=tf.stack([1, width])))
    x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
    if is_homogeneous:
        ones = tf.ones_like(x_t)
        coords = tf.stack([x_t, y_t, ones], axis=0)
    else:
        coords = tf.stack([x_t, y_t], axis=0)
    coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
    return coords


def projective_inverse_warp(img, depth, pose, intrinsics):
    """Inverse warp a source image to the target image plane based on projection.

  Args:
    img: the source image [batch, height_s, width_s, 3]
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source camera transformation matrix [batch, 6], in the
          order of tx, ty, tz, rx, ry, rz
    intrinsics: camera intrinsics [batch, 3, 3]
  Returns:
    Source image inverse warped to the target image plane [batch, height_t,
    width_t, 3]
  """
    batch, height, width, _ = img.get_shape().as_list()
    # Convert pose vector to matrix
    pose = pose_vec2mat(pose)
    # Construct pixel grid coordinates
    pixel_coords = meshgrid(batch, height, width)
    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
    # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, filler], axis=1)
    # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
    # pixel frame.
    proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
    output_img = bilinear_sampler(img, src_pixel_coords)
    return output_img


def bilinear_sampler(imgs, coords):
    """Construct a new image by bilinear sampling from the input image.

  Points falling outside the source image boundary have value 0.

  Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
  Returns:
    A new sampled image [batch, height_t, width_t, channels]
  """

    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = imgs.get_shape()
        coord_size = coords.get_shape()
        out_size = coords.get_shape().as_list()
        out_size[3] = imgs.get_shape().as_list()[3]

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        ## bilinear interp weights, with points outside the grid having weight 0
        # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
        # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
        # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
        # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

        wt_x0 = x1_safe - coords_x
        wt_x1 = coords_x - x0_safe
        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        ## indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        base = tf.reshape(
            _repeat(
                tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                coord_size[1] * coord_size[2]),
            [out_size[0], out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1])
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        ## sample from imgs
        imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
        imgs_flat = tf.cast(imgs_flat, 'float32')
        im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
        im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
        im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
        im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = tf.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11
        ])
        return output


def download(url, file_path):
    # 第一次请求是为了得到文件总大小
    r1 = requests.get(url, stream=True, verify=False)
    total_size = int(r1.headers['Content-Length'])

    # 这重要了，先看看本地文件下载了多少
    if os.path.exists(file_path):
        temp_size = os.path.getsize(file_path)  # 本地已经下载的文件大小
    else:
        temp_size = 0
    # 显示一下下载了多少
    print(temp_size)
    print(total_size)
    # 核心部分，这个是请求下载时，从本地文件已经下载过的后面下载
    headers = {'Range': 'bytes=%d-' % temp_size}
    # 重新请求网址，加入新的请求头的
    r = requests.get(url, stream=True, verify=False, headers=headers)

    # 下面写入文件也要注意，看到"ab"了吗？
    # "ab"表示追加形式写入文件
    with open(file_path, "ab") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                temp_size += len(chunk)
                f.write(chunk)
                f.flush()

                ###这是下载实现进度显示####
                done = int(50 * temp_size / total_size)
                sys.stdout.write("\r[%s%s] %d%%" % ('█' * done, ' ' * (50 - done), 100 * temp_size / total_size))
                sys.stdout.flush()
    print()  # 避免上面\r 回车符


def find_latest_ckpt(ckpt_path):
    '''
    查找ckpt_path目录下面最新的ckpt,
    如果存在latest,则返回model.latest.否则返回最新新的model-XXX
    :param ckpt_path:
    :return:
    '''
    dict_meta = {}
    for file in os.listdir(ckpt_path):
        if os.path.splitext(file)[1] == '.meta':
            prefix = os.path.splitext(file)[0]
            if prefix =='model.latest':
                return prefix
            else:
                dict_meta[prefix.split('-')[1]]=prefix
    return dict_meta[max(dict_meta, key=dict_meta.get)]


def evaluate_depth(pred_depths,test_file_list,kitti_dir,min_depth,max_depth):
    """
    evaluate the test result of depth net
    :param pred_file:
    :param test_file_list:
    :param kitti_dir:
    :param min_depth:
    :param max_depth:
    :return:
    """
    #pred_depths = np.load(pred_file)
    test_files = read_text_lines(test_file_list)
    gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, kitti_dir)
    num_test = len(im_files)
    gt_depths = []
    pred_depths_resized = []
    for t_id in range(num_test):
        camera_id = cams[t_id]  # 2 is left, 3 is right
        pred_depths_resized.append(
            cv2.resize(pred_depths[t_id],
                       (im_sizes[t_id][1], im_sizes[t_id][0]),
                       interpolation=cv2.INTER_LINEAR))
        depth = generate_depth_map(gt_calib[t_id],
                                   gt_files[t_id],
                                   im_sizes[t_id],
                                   camera_id,
                                   False,
                                   True)
        gt_depths.append(depth.astype(np.float32))
    pred_depths = pred_depths_resized

    rms = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel = np.zeros(num_test, np.float32)
    a1 = np.zeros(num_test, np.float32)
    a2 = np.zeros(num_test, np.float32)
    a3 = np.zeros(num_test, np.float32)
    for i in range(num_test):
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])

        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
        # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
        gt_height, gt_width = gt_depth.shape
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)

        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        # Scale matching
        scalor = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
        pred_depth[mask] *= scalor

        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
            compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10},  {:>10}"\
            .format('abs_rel', 'sq_rel', 'rms', 'log_rms',  'a1', 'a2', 'a3'))

    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}"\
          .format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(),a3.mean()))

    print("最终输出的值都是整个测试集上所有图片取均值之后的结果")
    print("abs_rel : (|gt-pred|/gt).mean")
    print("sq_rel  : ((gt-pred)^2/gt).mean")
    print("rms     : sqrt(((gt-pred)^2).mean)")
    print("log_rms : sqrt(((log(gt)-log(pred))^2).mean)")
    print("        thresh=np.maximum((gt / pred), (pred / gt))")
    print("a1      : (thresh<1.25  ).mean")
    print("a2      : (thresh<1.25^2).mean")
    print("a3      : (thresh<1.25^3).mean")
    pass


def evaluate_pose(pred_dir,gtruth_dir):
    """
    BUG:gt for pose evaluate is rely on  pose_eval_data.tar,
        we do not know the way used for generate the gt in the tar,
        sln 1:use 09 and 10 seq for test forever
        sln 2:find out the way to generate gt.
    :param pred_dir:
    :param gtruth_dir:
    :return:
    """
    pred_files = glob(pred_dir + '/*.txt')
    ate_all = []
    for i in range(len(pred_files)):
        gtruth_file = gtruth_dir + os.path.basename(pred_files[i])
        if not os.path.exists(gtruth_file):
            continue
        ate = compute_ate(gtruth_file, pred_files[i])
        if ate == False:
            continue
        ate_all.append(ate)
    ate_all = np.array(ate_all)
    print("Predictions dir: %s" % pred_dir)
    print("ATE(Absolute Trajectory Error,绝对轨迹误差) mean: %.4f, std: %.4f" % (np.mean(ate_all), np.std(ate_all)))
    pass

