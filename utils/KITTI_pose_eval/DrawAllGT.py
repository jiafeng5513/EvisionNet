import glob
import argparse
import os, os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import tools.transformations as tr

# choose other backend that not required GUI (Agg, Cairo, PS, PDF or SVG) when use matplotlib
# TkAgg is the default，use GUI
plt.switch_backend('TkAgg')
# 指定字体为SimHei，用于显示中文，如果Ariel,中文会乱码
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
# 用来正常显示负号
matplotlib.rcParams["axes.unicode_minus"] = False
def loadPoses(file_name):
    '''
        Each line in the file should follow one of the following structures
        (1) idx pose(3x4 matrix in terms of 12 numbers)
        (2) pose(3x4 matrix in terms of 12 numbers)
    '''
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    file_len = len(s)
    poses = {}
    frame_idx = 0
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split()]
        withIdx = int(len(line_split) == 13)
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row * 4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        poses[frame_idx] = P
    return poses


# GT轨迹在XZ
def plotPath_2D_GT(seq, poses_gt, plot_path_dir):
    fontsize_ = [10, 15, 10]
    plot_keys = ["Ground Truth", "EvisionNet"]
    start_point = [0, 0]
    style_gt = 'r-'
    style_O = 'ko'
    ### get the value
    if poses_gt:
        poses_gt = [(k, poses_gt[k]) for k in sorted(poses_gt.keys())]
        x_gt = np.asarray([pose[0, 3] for _, pose in poses_gt])
        y_gt = np.asarray([pose[1, 3] for _, pose in poses_gt])
        z_gt = np.asarray([pose[2, 3] for _, pose in poses_gt])

    fig = plt.figure(figsize=(16, 4), dpi=80)
    ### plot the figure
    plt.subplot(1, 4, 1)
    ax = plt.gca()
    if poses_gt:
        plt.plot(x_gt, z_gt, style_gt, label=plot_keys[0])
    plt.plot(start_point[0], start_point[1], style_O, label='起点')
    plt.legend(loc="upper left", prop={'size': fontsize_[0]})
    plt.xlabel('x (m)', fontsize=fontsize_[0])
    plt.ylabel('z (m)', fontsize=fontsize_[0])
    ### set the range of x and y
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean))
                       for lim in lims])
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])
    #plt.legend(loc='lower right', fontsize=fontsize_[0])  # 标签位置

    png_title = "{}path".format(seq)
    fig.tight_layout()

    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    seq = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    plot_path_dir = './ground_truth_pose/xz_plot/'
    gt_path = './ground_truth_pose/'
    for item in seq:
        pose_gt = loadPoses(gt_path+item+'.txt')
        plotPath_2D_GT(item,pose_gt,plot_path_dir)
        print(item)
    pass
