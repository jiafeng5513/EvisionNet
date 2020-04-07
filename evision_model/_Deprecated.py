# -*- coding: utf-8 -*-
"""
A bank for deprecated functions and modules.

code by jiafeng5513
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
"""
Randomized Layer Normalization

NOTE:
    1. 内部含有一个Batch Normalization
    2. 无论是否在训练模式下，内部的Batch Normalization都是启用的
    3. 内部的Batch Normalization使用自定义的均值（mean）和方差（variance）
    4. 训练时均值和方差乘以均值为1的高斯噪声
    5. 测试时使用真实的均值和方差
    6. see:https://arxiv.org/abs/1904.04998.
"""

def truncated_normal(shape, mean=0.0, stddev=1.0):
    """
    pytorch实现tf.random.truncated_normal
    see:https://zhuanlan.zhihu.com/p/83609874
    and
    https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16
    and
    https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/random/truncated_normal
    """
    tensor = torch.zeros(shape).cuda()
    tmp = tensor.new_empty(shape + (4,)).normal_().cuda()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(stddev).add_(mean)
    return tensor

def RandomizedLayerNormalization(input, is_train, stddev=0.5):
    mean = input.mean(axis=[0, 2, 3], keepdim=True)
    variance = input.var(axis=[0, 2, 3], keepdim=True)
    if is_train:
        mean *= 1.0 + truncated_normal(mean.shape, stddev=stddev)
        variance *= 1.0 + truncated_normal(variance.shape, stddev=stddev)
    #torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)
    outputs = torch.nn.functional.batch_norm(input=input, running_mean=mean, running_var=variance,  eps=1e-3)
    return outputs
    pass
