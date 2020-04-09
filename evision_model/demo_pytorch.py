import torch
import tensorflow.compat.v1 as tf
import numpy as np


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


if __name__ == '__main__':
    #demo_unique()
    input = torch.empty(3, 5, 7, 9)
    N,C,H,W = input.shape
    shape_list = [N,C,H,W]

    print(input.reshape([-1]).shape)