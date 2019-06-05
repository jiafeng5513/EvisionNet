# -*- coding: utf-8 -*-

import os
import zipfile
import shutil


root_dir = '/home/RAID1/DataSet/KITTI/KittiRaw/'
zip_dir = '/home/RAID1/DataSet/KITTI/KittiRaw_zip/'
subsets = ['2011_09_26','2011_09_28','2011_09_29','2011_09_30','2011_10_03']


def get_zip_file(input_path, result):
    """
    对目录进行深度优先遍历
    :param input_path:
    :param result:
    :return:
    """
    files = os.listdir(input_path)
    for file in files:
        if os.path.isdir(input_path + '/' + file):
            get_zip_file(input_path + '/' + file, result)
        else:
            result.append(input_path + '/' + file)

def zip_file_path(input_path, output_path, output_name):
    """
    压缩文件
    :param input_path: 压缩的文件夹路径
    :param output_path: 解压（输出）的路径
    :param output_name: 压缩包名称
    :return:
    """
    f = zipfile.ZipFile(output_path + '/' + output_name, 'w', zipfile.ZIP_DEFLATED)
    filelists = []
    get_zip_file(input_path, filelists)
    for file in filelists:
        f.write(file)
    # 调用了close方法才会保证完成压缩
    f.close()
    return output_path + r"/" + output_name

if __name__ == '__main__':
    for subset in subsets:
        # 在zip_dir建立文件夹
        if not os.path.exists(os.path.join(zip_dir,subset)):
            os.mkdir(os.path.join(zip_dir,subset))
        for item in os.listdir(os.path.join(root_dir , subset)):
            if os.path.isdir(os.path.join(root_dir, subset, item)):
                print('zip and trans:%s' % (item))
                zip_file_path(os.path.join(root_dir, subset, item), os.path.join(zip_dir,subset), item+'.zip')
            else:
                shutil.copyfile(os.path.join(root_dir, subset, item), os.path.join(zip_dir,subset,item))





