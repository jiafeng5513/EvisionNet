import torch
import tensorflow.compat.v1 as tf
import numpy as np
import torch.nn as nn

def demo_unique():
    x = tf.placeholder(tf.float32, shape=[9])
    y = tf.unique(x)[0]

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    x_vars = [1, 1, 2, 4, 4, 4, 7, 8, 8]

    print(sess.run(y, feed_dict={x: x_vars}))

    #y == > [1, 2, 4, 7, 8]
    #idx == > [0, 0, 1, 2, 2, 2, 3, 4, 4]
    pass

def demo_roll():
    # x = torch.Tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    # y = torch.roll(x, 1, 1)
    # print(y)# [[4., 0., 1., 2., 3.],[9., 5., 6., 7., 8.]]

    x = tf.placeholder(tf.float32, shape=[2,5])
    y = tf.roll(x, 1, 1)

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    x_vars = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    print(sess.run(y, feed_dict={x: x_vars}))


def demo_mul():
    x = torch.Tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    y = x.mul(x)
    print(y)


def dilate(x, foreground_dilation):
    # Dilation by n pixels is roughtly max pooling by 2 * n + 1.
    p = foreground_dilation * 2 + 1
    pool2d = nn.MaxPool2d((p, p), stride=(1, 1))  # TODO:tf.nn.max_pool(x, [1, p, p, 1], [1] * 4, 'SAME')
    return pool2d(x)


def dilate_tf(x):
    # Dilation by n pixels is roughtly max pooling by 2 * n + 1.
    p = 8 * 2 + 1
    return tf.nn.max_pool(x, [1, p, p, 1], [1,1,1,1], 'SAME')

def demo_dilate():
    # input = torch.empty(4, 1, 128, 416)
    # y = dilate(input, 8)
    # print (y.shape)  # [4, 1, 112, 400]

    x = tf.placeholder(tf.float32, shape=[4, 128, 416, 1])
    y = dilate_tf(x)

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    x_vars = np.random.rand(4, 128, 416, 1)

    sess.run(y, feed_dict={x: x_vars})
    print(y.shape)
    pass


if __name__ == '__main__':
    demo_dilate()
    # input = torch.empty(3, 5, 7, 9)
    # N,C,H,W = input.shape
    # shape_list = [N,C,H,W]
    #
    # print(input.reshape([-1]).shape)