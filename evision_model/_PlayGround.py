import torch
import tensorflow.compat.v1 as tf
import numpy as np
import torch.nn as nn


def tensorflow_test_template():
    x = tf.placeholder(tf.float32, shape=[4, 128, 416, 1])
    #y = function_to_be_test(x)

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    x_vars = np.random.rand(4, 128, 416, 1)

    r = sess.run(y, feed_dict={x: x_vars})
    #print(y.shape)
    #print(r)
    pass

def pytorch_test_template():
    image = torch.randn(4, 6, 128, 416)  # 输入尺寸 [N C H W]
    # function to be test
    pass

def item_tensor():
    image = torch.Tensor([[True,  True], [False, False]])  # 输入尺寸 [N C H W]
    y = image.sum(1,keepdim=True).bool()
    print(y)
    pass

def reduce_any_demo():
    x = tf.constant([[True, True], [False, False]])
    y1 = tf.reduce_any(x)  # True
    y2 = tf.reduce_any(x, 0)  # [True, True]
    y3 = tf.reduce_any(x, 1)  # [True, False]
    y4 = tf.reduce_any(x, 1, keepdims=True)
    sess = tf.Session()
    print(sess.run(y1))
    print(sess.run(y2))
    print(sess.run(y3))
    print(sess.run(y4))

    pass

def resample():
    from tensorflow.contrib.resampler import resampler
    img = tf.placeholder(tf.float32, shape=[1, 3, 3, 1])  # [n,h,w,c]
    pixel_xy = tf.placeholder(tf.float32, shape=[1, 3, 3, 2])  # [n,h,w,c]

    y = resampler(img, pixel_xy)

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    img_vars = [[[[0.1], [0.2], [0.3]],
                 [[0.4], [0.5], [0.6]],
                 [[0.7], [0.8], [0.9]]]]
    pixel_xy_vars = [[[[0.5, 0.5], [0.5, 1.5], [0.5, 2.5]],
                      [[1.5, 0.5], [1.5, 1.5], [1.5, 2.5]],
                      [[2.5, 0.5], [2.5, 1.5], [2.5, 2.5]]]]

    """img_vars
    [[ [[0.1] [0.2] [0.3]]
       [[0.4] [0.5] [0.6]]
       [[0.7] [0.8] [0.9]] ]]
    """

    """ pixel_xy_vars:
    [[ [[0.5 0.5] [0.5 1.5] [0.5 2.5]]
       [[1.5 0.5] [1.5 1.5] [1.5 2.5]]
       [[2.5 0.5] [2.5 1.5] [2.5 2.5]] ]]
    """

    """输出:
    [[ [[0.3       ] [0.6       ] [0.375     ]]
       [[0.39999998] [0.70000005] [0.425     ]]
       [[0.22500001] [0.375     ] [0.225     ]] ]]
    """
    r = sess.run(y, feed_dict={img: img_vars,pixel_xy:pixel_xy_vars})
    print(r)
    pass

def resample_torch():

    pass
if __name__ == '__main__':
    resample()
    pass
