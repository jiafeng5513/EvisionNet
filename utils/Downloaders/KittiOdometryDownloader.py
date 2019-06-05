# -*- coding: utf-8 -*-

import wget, tarfile
import os
import urllib
import tqdm
import sys
import requests

odometry=['https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip',  #  1 MB
          'https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip',      #  1 MB
          'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip',  #  4 MB
          'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip',   #  22 GB
          'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip',  #  65 GB
          'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip']#  80 GB

odometry_dir = 'H:/KITTI/KITTI Odometry/'

# 屏蔽warning信息
requests.packages.urllib3.disable_warnings()

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


if __name__ == '__main__':
    if not os.path.exists(odometry_dir):
        os.mkdir(odometry_dir)

    for item in odometry:
        filename=item[48:]
        print(odometry_dir+filename)
        download(item, odometry_dir+filename)
