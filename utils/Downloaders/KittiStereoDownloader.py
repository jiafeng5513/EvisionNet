# -*- coding: utf-8 -*-

import wget, tarfile
import os
import urllib
import tqdm
import sys
import requests


stereo2012=['https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip',
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow_calib.zip',
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow_multiview.zip',
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_stereo_flow.zip']

stereo2015=['https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip',
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_calib.zip',
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_multiview.zip',
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_scene_flow.zip']
stereo2012_dir = 'H:/KITTI/KITTI Stereo/Stereo 2012/'
stereo2015_dir = 'H:/KITTI/KITTI Stereo/Stereo 2015/'

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
    if not os.path.exists(stereo2012_dir):
        os.mkdir(stereo2012_dir)
    if not os.path.exists(stereo2015_dir):
        os.mkdir(stereo2015_dir)

    for item in stereo2012:
        filename=item[48:]
        print(stereo2012_dir+filename)
        # wget.download(item, out=stereo2012_dir+filename)
        download(item, stereo2012_dir+filename)

    for item in stereo2015:
        filename=item[48:]
        print(stereo2015_dir+filename)
        # wget.download(item, out=stereo2015_dir+filename)
        download(item, stereo2015_dir+filename)
