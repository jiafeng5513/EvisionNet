from pathlib import Path
import numpy as np
import csv
import re
import cv2


def read_calib(calib_file_path):
    with open(calib_file_path, 'r') as calib_file:
        calib = {}
        csv_reader = csv.reader(calib_file, delimiter='=')
        for attr, value in csv_reader:
            calib.setdefault(attr, value)

    return calib


def read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<'  # littel endian
            scale = -scale
        else:
            endian = '>'  # big endian

        dispariy = np.fromfile(pfm_file, endian + 'f')
    #
    img = np.reshape(dispariy, newshape=(height, width, channels))
    img = np.flipud(img).astype('uint8')
    #
    cv2.imwrite('./disparity.png',img)
    show(img, "disparity")

    return dispariy, [(height, width, channels), scale]


def create_depth_map(disparity, shape, scale, calib=None):
    if calib is None:
        raise Exception("Loss calibration information.")
    else:
        fx = float(calib['cam0'].split(' ')[0].lstrip('['))
        base_line = float(calib['baseline'])
        doffs = float(calib['doffs'])

        # scale factor is used here
        depth_map = fx * base_line / (disparity / scale + doffs)
        depth_map = np.reshape(depth_map, newshape=shape)
        depth_map = np.flipud(depth_map).astype('uint8')
        return depth_map


def show(img, win_name='image'):
    if img is None:
        raise Exception("Can't display an empty image.")
    else:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, img)
        # cv2.waitKey()
        # cv2.destroyWindow(win_name)


def main(path):
    pfm_file_dir = Path(path)
    calib_file_path = pfm_file_dir.joinpath('calib.txt')
    disp_left = pfm_file_dir.joinpath('disp0.pfm')
    # calibration information
    calib = read_calib(calib_file_path)
    disparity, [shape, scale] = read_pfm(disp_left)
    # create depth map
    depth_map_left = create_depth_map(disparity,shape, scale, calib)
    show(depth_map_left, "depth_map")
    img0 = cv2.imread(path+'im0.png')
    show(img0, "left")
    cv2.waitKey()


if __name__ == '__main__':
    main(r'../examples/Adirondack-perfect/')
