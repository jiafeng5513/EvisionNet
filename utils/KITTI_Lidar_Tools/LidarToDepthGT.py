# 使用KITTI激光雷达点云生成深度图的GroundTruth
# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
from collections import Counter

from imageio import imwrite
from path import Path
from scipy.misc import imread
from tqdm import tqdm
import datetime
import os
import cv2
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1
    return points


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


def generate_depth_map(calib_dir, velo_file_name, im_shape, cam=2):
    # load calibration files
    cam2cam = read_calib_file(calib_dir + '/calib_cam_to_cam.txt')
    velo2cam = read_calib_file(calib_dir + '/calib_velo_to_cam.txt')
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, -1:]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0
    return depth


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow    BGR
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )
    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy() / max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert (tensor.size(0) == 3)
        array = 0.5 + tensor.numpy() * 0.5
    return array


def demo():
    map = [[0.0, 0.0, 0.0, 0.0],
           [0.5, 0.5, 0.5, 0.5],
           [0.7, 0.7, 0.7, 0.7],
           [0.9, 0.9, 0.9, 0.9]]
    array = COLORMAPS['rainbow'](map).astype(np.float32)
    pass


if __name__ == '__main__':
    demo()
    # 1. 从kitti_raw挑选要用的测试图片,将对应的png和bin复制到一个文件夹下,注意不要混用不同序列
    test_input_dir = 'H:/data/KITTI/TestDepth-RAW-09-26'
    # 2. 测试图片所在序列的标定文件所在目录
    calib_file_path = 'H:/data/KITTI/KittiRaw/2011_09_26'
    # 3.
    f_list = os.listdir(test_input_dir)
    bin_file_list = []
    for i in f_list:
        if os.path.splitext(i)[1] == '.bin':
            bin_file_list.append(test_input_dir + '/' + i)
    cv2.namedWindow('GT')
    for bin_file in bin_file_list:
        image_file = str(os.path.splitext(bin_file)[0]) + '.png'
        tgt = imread(image_file).astype(np.float32)
        im_shape = tgt.shape[:2]
        output = generate_depth_map(calib_file_path, bin_file, im_shape, cam=2)

        array = output
        max_value = np.max(array)
        array = array / max_value
        # 插值
        from scipy.interpolate import Rbf

        x = []
        y = []
        z = []
        XI = np.arange(array.shape[1])
        YI = np.arange(array.shape[0])

        XX,YY = np.meshgrid(XI,YI)

        for i in XI:  # 1242
            for j in YI:  # 375
                if array[j][i] != 0:
                    x.append(i)
                    y.append(j)
                    z.append(array[j][i])
        rbf = Rbf(x, y, z, epsilon=2)  # 这是坐标点和像素值
        ZZ = rbf(XX, YY)  # 参数是插值范围

        # plot the result
        import matplotlib.pyplot as plt

        plt.subplot(1, 1, 1)
        plt.pcolor(XX, YY, ZZ, cmap=cm.jet)
        plt.scatter(x, y, 100, z, cmap=cm.jet)
        plt.title('RBF interpolation - multiquadrics')
        plt.xlim(0, 1241)
        plt.ylim(0, 374)
        plt.colorbar()

        array = COLORMAPS['rainbow'](array).astype(np.float32)
        array = array.transpose(0, 1, 2)
        array = (255 * array).astype(np.uint8)
        array = np.transpose(array, (0, 1, 2))

        img = array
        # imwrite(str(os.path.splitext(bin_file)[0])+'_gt.png', img)

        cv2.imshow('GT', img)
        cv2.waitKey(0)
        # print(depth.shape)
    pass
# import os
# import matplotlib.pyplot as plt
#
# plt.figure("Image")  # 图像窗口名称
# plt.imshow(img)
# plt.axis('on')  # 关掉坐标轴为 off
# plt.title('image')  # 图像题目
# plt.show()
