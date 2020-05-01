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


def resample_tf():
    from tensorflow.contrib.resampler import resampler
    import time
    img = tf.placeholder(tf.float32, shape=[1, 5, 5, 1])  # [n,h,w,c]
    pixel_xy = tf.placeholder(tf.float32, shape=[1, 5, 5, 2])  # [n,h,w,c]

    y = resampler(img, pixel_xy)

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    img_vars = np.array([[[[0.1], [0.2], [0.3], [0.4], [0.5]],
                          [[0.6], [0.7], [0.8], [0.9], [1.0]],
                          [[1.1], [1.2], [1.3], [1.4], [1.5]],
                          [[1.6], [1.7], [1.8], [1.9], [2.0]],
                          [[2.1], [2.2], [2.3], [2.4], [2.5]]]])
    pixel_xy_vars = np.array([[[[0.5, 0.5], [0.5, 1.5], [0.5, 2.5], [0.5, 3.5], [0.5, 4.5]],
                               [[1.5, 0.5], [1.5, 1.5], [1.5, 2.5], [1.5, 3.5], [1.5, 4.5]],
                               [[2.5, 0.5], [2.5, 1.5], [2.5, 2.5], [2.5, 3.5], [2.5, 4.5]],
                               [[3.5, 0.5], [3.5, 1.5], [3.5, 2.5], [3.5, 3.5], [3.5, 4.5]],
                               [[4.5, 0.5], [4.5, 1.5], [4.5, 2.5], [4.5, 3.5], [4.5, 4.5]]]])
    start = time.time()
    r = sess.run(y, feed_dict={img: img_vars, pixel_xy: pixel_xy_vars})
    end = time.time()
    print("tensorflow.contrib.resampler done in %.5f s" % (end - start))
    print(r)
    """
    [[[[0.400], [0.900], [1.400], [1.900], [1.075]],
      [[0.500], [1.000], [1.500], [2.000], [1.125]],
      [[0.600], [1.100], [1.600], [2.100], [1.175]],
      [[0.700], [1.200], [1.700], [2.200], [1.225]],
      [[0.375], [0.625], [0.875], [1.125], [0.625]]]]
    """
    print("===============end of tf resampler================")
    pass


def resample_new(int_map=None, xy_float=None):
    """
        int_map上的像素处于整数网格坐标下,xy_float描述了一个浮点数网格
        该函数的作用就是把int_map在xy_float下进行重新采样
    Args:
        int_map      :   输入张量, [B,C,H_in,W_in],C可能是1或者3
        xy_float     :   浮点数网格坐标,[B,H_out,W_out,2],需要注意,这是坐标网格,不遵守[b.c.h.w]
    Returns:
        调整后的数据  :   [B,C,H_out,W_out]
    """
    b_1, c_1, H_in, W_in = int_map.shape
    b_2, H_out, W_out, c_2 = xy_float.shape

    # 把坐标归一化到[-1,1],数据类型保持在float
    x_float, y_float = torch.unbind(xy_float, dim=3)  # 把x和y 拆看,分别广播操作
    x_float = torch.unsqueeze((2 * x_float / (W_in - 1)) - 1, dim=-1)  # new_x = 2*x/(w-1)-1
    y_float = torch.unsqueeze((2 * y_float / (H_in - 1)) - 1, dim=-1)  # new_y = 2*y/(h-1)-1
    grid = torch.cat((x_float, y_float), dim=-1).float()
    output = torch.nn.functional.grid_sample(int_map.float(), grid,
                                             mode='bilinear', padding_mode='zeros', align_corners=None)
    return output


def less_tf():
    x = tf.placeholder(tf.float32, shape=[4])
    y = tf.placeholder(tf.float32, shape=[4])

    z = tf.less(x, y)

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    x_vars = [1, 2, 3, 4]
    y_vars = [0, 1, 5, 6]

    r = sess.run(z, feed_dict={x: x_vars, y: y_vars})
    print(r)
    pass


def less_troch():
    x = torch.from_numpy(np.array([1, 2, 3, 4]))
    y = torch.from_numpy(np.array([0, 1, 5, 6]))

    z = torch.le(x, y)

    print(z)
    pass


def Broadcast_tf():
    x = tf.placeholder(tf.float32, shape=[4, 1, 1, 3])
    y = tf.placeholder(tf.float32, shape=[4, 5, 7, 3])

    z = tf.broadcast_to(x, y.shape)

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    x_vars = np.random.rand(4, 1, 1, 3)
    y_vars = np.random.rand(4, 5, 7, 3)
    r = sess.run(z, feed_dict={x: x_vars, y: y_vars})
    print(z.shape)
    pass

def Broadcast_torch():
    x = torch.from_numpy(np.random.rand(4, 1, 1, 3))
    y = torch.from_numpy(np.random.rand(4, 5, 7, 3))

    z, _ = torch.broadcast_tensors(x, y)
    print(z.shape)
    pass


def batch_eye_tf():
    y = tf.eye(3, batch_shape=[4])
    print(y)
    pass

def batch_eye_torch():
    y = torch.eye(3).unsqueeze(0).repeat(4, 1, 1)
    print(y.shape)
    print(y)
    pass

if __name__ == '__main__':
    # batch_eye_tf()
    batch_eye_torch()
    pass
