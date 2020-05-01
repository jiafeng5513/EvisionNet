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


# 太慢
def resample_scipy(int_map=None, xy_float=None):
    """
        int_map上的像素处于整数网格坐标下,xy_float描述了一个浮点数网格
        该函数的作用就是把int_map在xy_float下进行重新采样
    Args:
        int_map   :   输入张量, [B,C,H,W],C可能是1或者3
        xy_float  :   浮点数网格坐标,[B,2,H,W]
    Returns:
        调整后的数据
    """

    import numpy as np
    from scipy import interpolate

    # 先确定尺寸
    batch_size, channels, h, w = int_map.shape
    if channels != 3 and channels != 1:
        raise ValueError('int_map tensor must be [B,1,H,W] or [B,3,H,W], but [%d,%d,%d,%d] is provided.' %
                         (batch_size, channels, h, w))
    # 建立标准整型网格
    x_normal, y_normal = np.mgrid[0:h, 0:w]
    # 拆分张量,准备进行batch_size次迭代
    x_float, y_float = torch.unbind(xy_float, dim=1)  # 拆开x,y
    list_x = torch.unbind(x_float, dim=0)  # 拆开Batch
    list_y = torch.unbind(y_float, dim=0)  # 拆开Batch
    list_input = torch.unbind(int_map, dim=0)  # 拆开Batch
    v_List = []
    for v, x, y in zip(list_input, list_x, list_y):  # v,x,y都是张量
        _x = x.numpy()[:, 0]  # 提取列
        _y = y.numpy()[0]  # 提取行
        if v.shape[0] == 1:  # 单通道
            interfunc = interpolate.interp2d(x_normal, y_normal, v.squeeze().numpy(), kind='cubic')
            v_List.append(torch.from_numpy(interfunc(_x, _y)))
        elif v.shape[0] == 3:  # 三通道
            r, g, b = torch.unbind(v, dim=0)  # 拆开三个通道
            interfunc_r = interpolate.interp2d(x_normal, y_normal, r.numpy(), kind='cubic')
            interfunc_g = interpolate.interp2d(x_normal, y_normal, g.numpy(), kind='cubic')
            interfunc_b = interpolate.interp2d(x_normal, y_normal, b.numpy(), kind='cubic')
            v_List.append(torch.cat((torch.from_numpy(interfunc_r(_x, _y)),
                                     torch.from_numpy(interfunc_g(_x, _y)),
                                     torch.from_numpy(interfunc_b(_x, _y))), dim=0))
    result = torch.cat(tuple(v_List), dim=0)
    return result


# 有数据
def resample_test():
    import numpy as np
    from scipy import interpolate
    x = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0],
                  [2.0, 2.0, 2.0, 2.0, 2.0],
                  [3.0, 3.0, 3.0, 3.0, 3.0],
                  [4.0, 4.0, 4.0, 4.0, 4.0]])
    y = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                  [0.0, 1.0, 2.0, 3.0, 4.0],
                  [0.0, 1.0, 2.0, 3.0, 4.0],
                  [0.0, 1.0, 2.0, 3.0, 4.0],
                  [0.0, 1.0, 2.0, 3.0, 4.0]])
    v = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                  [0.6, 0.7, 0.8, 0.9, 1.0],
                  [1.1, 1.2, 1.3, 1.4, 1.5],
                  [1.6, 1.7, 1.8, 1.9, 2.0],
                  [2.1, 2.2, 2.3, 2.4, 2.5]])
    # v3 = np.array([[[0.1,0.1,0.1], [0.2,0.2,0.2], [0.3,0.3,0.3], [0.4,0.4,0.4], [0.5,0.5,0.5]],
    #               [[0.6,0.6,0.6], [0.7,0.7,0.7], [0.8,0.8,0.8], [0.9,0.9,0.9], [1.0,1.0,1.0]],
    #               [[1.1,1.1,1.1], [1.2,1.2,1.2], [1.3,1.3,1.3], [1.4,1.4,1.4], [1.5,1.5,1.5]],
    #               [[1.6,1.6,1.6], [1.7,1.7,1.7], [1.8,1.8,1.8], [1.9,1.9,1.9], [2.0,2.0,2.0]],
    #               [[2.1,2.1,2.1], [2.2,2.2,2.2], [2.3,2.3,2.3], [2.4,2.4,2.4], [2.5,2.5,2.5]]])

    # 三次样条二维插值
    newfunc = interpolate.interp2d(x, y, v, kind='cubic')

    #
    xnew = np.array([0.5, 1.5, 2.5, 3.5, 4.5])  # x
    ynew = np.array([0.5, 1.5, 2.5, 3.5, 4.5])  # y
    v_new = newfunc(xnew, ynew)  # 仅仅是y值 100*100的值

    print(v_new)
    """
    [[0.4  0.9  1.4  1.9  2.15]
     [0.5  1.   1.5  2.   2.25]
     [0.6  1.1  1.6  2.1  2.35]
     [0.7  1.2  1.7  2.2  2.45]
     [0.75 1.25 1.75 2.25 2.5 ]]
    """
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


def resample_pytorch():
    import torch.nn.functional as F
    # theta = torch.tensor([
    #     [1, 0, -0.1],
    #     [0, 1, -0.1]
    # ], dtype=torch.float)
    # grid = F.affine_grid(theta.unsqueeze(0), torch.Size([1, 1, 5, 5]))
    # print(grid)
    """
    [1, 0, -0.5],
    [0, 1, -0.5]
    [[[[-1.5, -1.5], [-1.0, -1.5], [-0.5, -1.5], [ 0.0, -1.5], [ 0.5, -1.5]],
      [[-1.5, -1.0], [-1.0, -1.0], [-0.5, -1.0], [ 0.0, -1.0], [ 0.5, -1.0]],
      [[-1.5, -0.5], [-1.0, -0.5], [-0.5, -0.5], [ 0.0, -0.5], [ 0.5, -0.5]],
      [[-1.5,  0.0], [-1.0,  0.0], [-0.5,  0.0], [ 0.0,  0.0], [ 0.5,  0.0]],
      [[-1.5,  0.5], [-1.0,  0.5], [-0.5,  0.5], [ 0.0,  0.5], [ 0.5,  0.5]]]]
      
      [1, 0, 0],
      [0, 1, 0]
      [[[[-1.0, -1.0],[-0.5, -1.0],[ 0.0, -1.0],[ 0.5, -1.0],[ 1.0, -1.0]],
        [[-1.0, -0.5],[-0.5, -0.5],[ 0.0, -0.5],[ 0.5, -0.5],[ 1.0, -0.5]],
        [[-1.0,  0.0],[-0.5,  0.0],[ 0.0,  0.0],[ 0.5,  0.0],[ 1.0,  0.0]],
        [[-1.0,  0.5],[-0.5,  0.5],[ 0.0,  0.5],[ 0.5,  0.5],[ 1.0,  0.5]],
        [[-1.0,  1.0],[-0.5,  1.0],[ 0.0,  1.0],[ 0.5,  1.0],[ 1.0,  1.0]]]]
    """
    grid = torch.from_numpy(np.array([[[[-0.75, -0.75], [-0.75, -0.25], [-0.75, 0.25], [-0.75, 0.75], [-0.75, 1.25]],
                                       [[-0.25, -0.75], [-0.25, -0.25], [-0.25, 0.25], [-0.25, 0.75], [-0.25, 1.25]],
                                       [[0.25, -0.75], [0.25, -0.25], [0.25, 0.25], [0.25, 0.75], [0.25, 1.25]],
                                       [[0.75, -0.75], [0.75, -0.25], [0.75, 0.25], [0.75, 0.75], [0.75, 1.25]],
                                       [[1.25, -0.75], [1.25, -0.25], [1.25, 0.25], [1.25, 0.75],
                                        [1.25, 1.25]]]])).float()
    img = torch.from_numpy(np.array([[[[0.1, 0.2, 0.3, 0.4, 0.5],
                                       [0.6, 0.7, 0.8, 0.9, 1.0],
                                       [1.1, 1.2, 1.3, 1.4, 1.5],
                                       [1.6, 1.7, 1.8, 1.9, 2.0],
                                       [2.1, 2.2, 2.3, 2.4, 2.5]]]])).float()
    start = time.time()
    output = F.grid_sample(img, grid)
    end = time.time()
    print("grid_sample done in %.5f s" % (end - start))
    print(output)
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
    # 先把坐标归一化到[-1,1],数据类型保持在float
    # new_x = 2*x/(w-1)-1
    # new_y = 2*y/(h-1)-1
    import torch.nn.functional as F
    b_1, c_1, H_in, W_in = int_map.shape
    b_2, H_out, W_out, c_2 = xy_float.shape

    # 把x和y 拆看,分别广播操作
    x_float, y_float = torch.unbind(xy_float, dim=3)
    x_float = torch.unsqueeze((2 * x_float / (W_in - 1)) - 1, dim=-1)
    y_float = torch.unsqueeze((2 * y_float / (H_in - 1)) - 1, dim=-1)
    grid = torch.cat((x_float, y_float), dim=-1)
    output = F.grid_sample(int_map, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
    return output
    pass


if __name__ == '__main__':
    # resample_tf()
    # resample_pytorch()
    img = torch.from_numpy(np.array([[[[0.1, 0.2, 0.3, 0.4, 0.5],
                                       [0.6, 0.7, 0.8, 0.9, 1.0],
                                       [1.1, 1.2, 1.3, 1.4, 1.5],
                                       [1.6, 1.7, 1.8, 1.9, 2.0],
                                       [2.1, 2.2, 2.3, 2.4, 2.5]]]])).float()
    xy_float = torch.from_numpy(np.array([[[[0.5, 0.5], [0.5, 1.5], [0.5, 2.5], [0.5, 3.5], [0.5, 4.5]],
                                           [[1.5, 0.5], [1.5, 1.5], [1.5, 2.5], [1.5, 3.5], [1.5, 4.5]],
                                           [[2.5, 0.5], [2.5, 1.5], [2.5, 2.5], [2.5, 3.5], [2.5, 4.5]],
                                           [[3.5, 0.5], [3.5, 1.5], [3.5, 2.5], [3.5, 3.5], [3.5, 4.5]],
                                           [[4.5, 0.5], [4.5, 1.5], [4.5, 2.5], [4.5, 3.5], [4.5, 4.5]]]])).float()

    out = resample_new(img, xy_float)
    print(out)
    pass
