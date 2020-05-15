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


if __name__ == '__main__':
    # batch_eye_tf()
    dataflow_test()
    pass
