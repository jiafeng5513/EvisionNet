import torch
import numpy as np
import argparse

from imageio import imread, imsave
from scipy.misc import imresize
from path import Path
from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from models import DepthNet

parser = argparse.ArgumentParser(description='Inference script for DepthNet on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--output-disp", action='store_true', help="保存视差图")
parser.add_argument("--output-depth", action='store_true', help="保存深度图")

parser.add_argument("--pretrained", required=True, type=str, help="预训练DepthNet模型文件路径")
parser.add_argument("--img-height", default=128, type=int, help="输入图片的高度")
parser.add_argument("--img-width", default=416, type=int, help="输入图片的宽度")
parser.add_argument("--no-resize", action='store_true', help="禁用resize")

parser.add_argument("--dataset-list", default=None, type=str, help="测试图片文件列表文件")
parser.add_argument("--dataset-dir", default='.', type=str, help="测试图片的所在路径")
parser.add_argument("--output-dir", default='output', type=str, help="输出路径")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="图像文件扩展名")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    if not (args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return

    depth_net = DepthNet().to(device)
    weights = torch.load(args.pretrained)
    depth_net.load_state_dict(weights['state_dict'])
    depth_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir / file for file in f.read().splitlines()]
    else:
        test_files = sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])

    print('{} files to test'.format(len(test_files)))

    for file in tqdm(test_files):

        img = imread(file).astype(np.float32)

        h, w, _ = img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))

        tensor_img = torch.from_numpy(img).unsqueeze(0)
        tensor_img = ((tensor_img / 255 - 0.5) / 0.2).to(device)

        output = depth_net(tensor_img)[0]

        file_path, file_ext = file.relpath(args.dataset_dir).splitext()
        file_name = '-'.join(file_path.splitall())

        if args.output_disp:
            disp = (255 * tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
            imsave(output_dir / '{}_disp{}'.format(file_name, file_ext), np.transpose(disp, (1, 2, 0)))
        if args.output_depth:
            depth = 1 / output
            depth = (255 * tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
            imsave(output_dir / '{}_depth{}'.format(file_name, file_ext), np.transpose(depth, (1, 2, 0)))


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


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


if __name__ == '__main__':
    main()
