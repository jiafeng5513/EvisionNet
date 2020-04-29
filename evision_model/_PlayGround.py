import torch
import tensorflow.compat.v1 as tf
import numpy as np
import torch.nn as nn


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


def pytorch_test_template():
    image = torch.randn(4, 6, 128, 416)  # 输入尺寸 [N C H W]
    # function to be test
    pass


def item_tensor():
    image = torch.Tensor([[True, True], [False, False]])  # 输入尺寸 [N C H W]
    y = image.sum(1, keepdim=True).bool()
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
    img = tf.placeholder(tf.float32, shape=[1, 5, 5, 1])  # [n,h,w,c]
    pixel_xy = tf.placeholder(tf.float32, shape=[1, 5, 5, 2])  # [n,h,w,c]

    y = resampler(img, pixel_xy)

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    img_vars = [[[[0.1], [0.2], [0.3], [0.4], [0.5]],
                 [[0.6], [0.7], [0.8], [0.9], [1.0]],
                 [[1.1], [1.2], [1.3], [1.4], [1.5]],
                 [[1.6], [1.7], [1.8], [1.9], [2.0]],
                 [[2.1], [2.2], [2.3], [2.4], [2.5]]]]
    pixel_xy_vars = [[[[0.5, 0.5], [0.5, 1.5], [0.5, 2.5], [0.5, 3.5], [0.5, 4.5]],
                      [[1.5, 0.5], [1.5, 1.5], [1.5, 2.5], [1.5, 3.5], [1.5, 4.5]],
                      [[2.5, 0.5], [2.5, 1.5], [2.5, 2.5], [2.5, 3.5], [2.5, 4.5]],
                      [[3.5, 0.5], [3.5, 1.5], [3.5, 2.5], [3.5, 3.5], [3.5, 4.5]],
                      [[4.5, 0.5], [4.5, 1.5], [4.5, 2.5], [4.5, 3.5], [4.5, 4.5]]]]

    """img_vars
      [[ [[0.1], [0.2], [0.3], [0.4], [0.5]],
         [[0.6], [0.7], [0.8], [0.9], [1.0]],
         [[1.1], [1.2], [1.3], [1.4], [1.5]],
         [[1.6], [1.7], [1.8], [1.9], [2.0]],
         [[2.1], [2.2], [2.3], [2.4], [2.5]] ]]
    """

    """ pixel_xy_vars:
    [[ [[0.5, 0.5], [0.5, 1.5], [0.5, 2.5], [0.5, 3.5], [0.5, 4.5]],
       [[1.5, 0.5], [1.5, 1.5], [1.5, 2.5], [1.5, 3.5], [1.5, 4.5]],
       [[2.5, 0.5], [2.5, 1.5], [2.5, 2.5], [2.5, 3.5], [2.5, 4.5]],
       [[3.5, 0.5], [3.5, 1.5], [3.5, 2.5], [3.5, 3.5], [3.5, 4.5]],
       [[4.5, 0.5], [4.5, 1.5], [4.5, 2.5], [4.5, 3.5], [4.5, 4.5]] ]]
    """

    """输出:
    [[ [[0.40000004], [0.90000004], [1.4000001 ], [1.9000001 ], [1.075     ]],
       [[0.5       ], [1.        ], [1.5       ], [2.        ], [1.125     ]],
       [[0.6       ], [1.1       ], [1.6       ], [2.1       ], [1.175     ]],
       [[0.7       ], [1.2       ], [1.7       ], [2.2       ], [1.225     ]],
       [[0.375     ], [0.625     ], [0.875     ], [1.125     ], [0.625     ]] ]]
    """
    r = sess.run(y, feed_dict={img: img_vars, pixel_xy: pixel_xy_vars})
    print(r)
    pass


def resample_torch():
    from torch.nn.functional import grid_sample
    """img_vars[B,H,W,C]->[B,C,H,W] : [1, 3, 3, 1]->[1, 1, 3, 3]
    [[[[ 0.1,  0.2,  0.3],
       [ 0.4,  0.5,  0.6],
       [ 0.7,  0.8,  0.9]]]]
    """

    """ pixel_xy_vars:
    [[ [[0.5 0.5] [0.5 1.5] [0.5 2.5]]
       [[1.5 0.5] [1.5 1.5] [1.5 2.5]]
       [[2.5 0.5] [2.5 1.5] [2.5 2.5]] ]]
       
    [[[[0.7000, 0.4250, 0.0000],
       [0.3750, 0.2250, 0.0000],
       [0.0000, 0.0000, 0.0000]]]]

    """
    image = torch.Tensor([[[[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6],
                            [0.7, 0.8, 0.9]]]])
    pixel_xy = torch.Tensor([[[[1.5, 1.5], [1.5, 2.5], [1.5, 3.5]],
                              [[2.5, 1.5], [2.5, 2.5], [2.5, 3.5]],
                              [[3.5, 1.5], [3.5, 2.5], [3.5, 3.5]]]])
    y = grid_sample(input=image, grid=pixel_xy, mode='bilinear', padding_mode='reflection', align_corners=None)
    print(y)
    pass


def resample_my(input=None, xy_float=None):
    """
        将整数坐标调整到浮点数坐标
    [b,h,w](batch, height, width)形式的张量:pixel_x, pixel_y, depth
    Args:
        input     :   输入张量, [B,C,H,W],C 可能是1或者3
        xy_float  :   浮点数网格坐标,[B,2,H,W]
    Returns:
        调整后的数据张量
    """

    import numpy as np
    from scipy import interpolate

    # 先确定尺寸
    batch_size, channels, h, w = input.shape
    # 建立标准整型网格
    x_normal, y_normal = np.mgrid[0:h, 0:w]

    x_float, y_float = torch.unbind(xy_float, dim=3)  # 拆开x,y
    list_x = torch.unbind(x_float, dim=0)  # 拆开Batch
    list_y = torch.unbind(y_float, dim=0)  # 拆开Batch
    list_input = torch.unbind(input, dim=0)  # 拆开Batch

    vList = []
    for v, x, y in (list_input, list_x, list_y):
        if v.shape[0] == 1:  # 单通道
            # 拆掉"1"通道
            interfunc = interpolate.interp2d(x_normal, y_normal, v.squeeze().numpy(), kind='cubic')
            vList.append(interfunc(x, y))  # 这里不对
            pass
        elif item.shape[0] == 3:  # 三通道
            pass
        else:
            pass
    result = torch.cat(vList, dim=0)
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


    # 三次样条二维插值
    newfunc = interpolate.interp2d(x, y, v, kind='cubic')

    #
    xnew = np.array([0.5, 1.5, 2.5, 3.5, 4.5])  # x
    ynew = np.array([0.5, 1.5, 2.5, 3.5, 4.5])  # y
    v_new = newfunc(xnew, ynew)  # 仅仅是y值 100*100的值

    print(v_new)

    pass

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
    # v = np.array([[[0.1,0.1,0.1], [0.2,0.2,0.2], [0.3,0.3,0.3], [0.4,0.4,0.4], [0.5,0.5,0.5]],
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
if __name__ == '__main__':
    # resample()
    resample_my()
    # image = torch.randn(1, 3, 3, 2)
    # print(image)
    pass
