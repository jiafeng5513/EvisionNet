"""
定义常用常量
"""
import platform

Table = {}


# 平台相关常量

def initConstTable():
    print('Now running at %s' % platform.system())
    if platform.system() == 'Windows':
        Table['KittiRaw_formatted'] = 'G:/data/KITTI/KittiRaw_formatted/'
        Table['KittiRaw'] = 'G:/data/KITTI/KittiRaw/'
        Table['KittiOdometry'] = 'G:/data/KITTI/KittiOdometry/'
    elif platform.system() == 'Linux':
        Table['KittiRaw_formatted'] = '/home/RAID1/DataSet/KITTI/KittiRaw_formatted/'
        Table['KittiRaw'] = '/home/RAID1/DataSet/KITTI/KittiRaw/'
        Table['KittiOdometry'] = '/home/RAID1/DataSet/KITTI/KittiOdometry/'
    else:
        print('Evision CAN NOT running at current OS!')
    pass
