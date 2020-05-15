import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence DataFlow loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []  # 把所有的图片组织称样本序列,存到这个里面
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:  # 每个场景有自己的内参
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))  # 场景中所有的图片
            if len(imgs) < sequence_length:  # 如果这个场景的素有图片装不满一个样本,只能跳过这个场景
                continue
            for i in range(len(imgs)-sequence_length+1):
                sample = {'intrinsics': intrinsics, 'imgs':  []}
                for j in range(sequence_length):
                    sample['imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [load_as_float(img) for img in sample['imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform(imgs, np.copy(sample['intrinsics']))
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return imgs, intrinsics

    def __len__(self):
        return len(self.samples)
