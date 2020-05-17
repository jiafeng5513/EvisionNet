import torch
import tensorflow.compat.v1 as tf
import numpy as np
import torch.nn as nn
import time


# tf模板
def tensorflow_test_template():
    x = tf.placeholder(tf.float32, shape=[4, 128, 416, 1])
    # y = function_to_be_test(x)

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    x_vars = np.random.rand(4, 128, 416, 1)

    # r = sess.run(y, feed_dict={x: x_vars})
    # print(y.shape)
    # print(r)
    pass


# pytorch模板
def pytorch_test_template():
    image = torch.randn(4, 6, 128, 416)  # 输入尺寸 [N C H W]
    # function to be test
    pass


def dataflow_test():
    from DataFlow.sequence_folders import SequenceFolder
    import custom_transforms
    from torch.utils.data import DataLoader

    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([custom_transforms.RandomHorizontalFlip(),
                                                 custom_transforms.RandomScaleCrop(),
                                                 custom_transforms.ArrayToTensor(), normalize])
    datapath = 'G:/data/KITTI/KittiRaw_formatted'
    seed = 8964
    train_set = SequenceFolder(datapath, transform=train_transform, seed=seed, train=True,
                               sequence_length=3)

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4,
                              pin_memory=True)

    print("length of train loader is %d" % len(train_loader))

    dataiter = iter(train_loader)
    imgs, intrinsics = next(dataiter)
    print(len(imgs))
    print(intrinsics.shape)

    pass


def tf_grid():
    grid = tf.squeeze(tf.stack(tf.meshgrid(tf.range(4), tf.range(3), (1,))), axis=3)
    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    r = sess.run(grid)
    print(grid.shape)  # (3, 3, 4)
    print(r)
    """
    [4,3,3] [3,128,416] []
    [[[0 1 2 3] [0 1 2 3] [0 1 2 3]]
     [[0 0 0 0] [1 1 1 1] [2 2 2 2]]
     [[1 1 1 1] [1 1 1 1] [1 1 1 1]]]
    """
    pass


def torch_grid():
    grid = torch.stack(torch.meshgrid(torch.arange(start=0, end=4), torch.arange(start=0, end=3)))
    print(grid)
    """
    [[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
     [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]])
    """
    pass


def tf_perm():
    rank = 4
    perm = tf.concat([tf.range(rank - 1), [rank], [rank - 1]], axis=0)
    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    r = sess.run(perm)
    print(perm.shape)
    print(r)
    pass


if __name__ == '__main__':
    # tf_perm()
    a = torch.randn(4, 128, 416, 3, 3)
    x = torch.randn(4, 128, 416, 3, 3)
    x.permute()
    y = torch.matmul(a, x)

    print(y.shape)

    pass
