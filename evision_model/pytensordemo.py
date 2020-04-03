import torch
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers
import numpy as np


def pytorch_demo():
    image = torch.randn(1, 4, 128, 416)  # 输入尺寸
    image2 = torch.randn(1, 4, 128, 416)  # 输入尺寸

    out = torch.cat([image, image2], dim=-1)
    # N, C, H, W = layer.shape
    print(out.shape)
    pass


def tensorflow_demo():
    x = tf.placeholder(tf.float32, shape=[1, 128, 416, 6])
    weight_reg = 0.0
    y = layers.conv2d(
        x,
        16, [3, 3],
        stride=2,
        scope='Conv1',
        weights_regularizer=layers.l2_regularizer(weight_reg),
        activation_fn=tf.nn.relu)
    y = layers.conv2d(
        y,
        32, [3, 3],
        stride=2,
        scope='Conv2',
        weights_regularizer=layers.l2_regularizer(weight_reg),
        activation_fn=tf.nn.relu)
    y = layers.conv2d(
        y,
        64, [3, 3],
        stride=2,
        scope='Conv3',
        weights_regularizer=layers.l2_regularizer(weight_reg),
        activation_fn=tf.nn.relu)

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    x_vars = np.random.rand(1, 128, 416, 6)
    # print(x_vars)
    sess.run(y, feed_dict={x: x_vars})
    print(y.shape)
    pass


def _refine_motion_field(motion_field, layer):
    _, h, w, _ = tf.unstack(tf.shape(layer))
    upsampled_motion_field = tf.image.resize_bilinear(motion_field, [h, w])
    conv_input = tf.concat([upsampled_motion_field, layer], axis=3)
    conv_output = layers.conv2d(
        conv_input, max(4, layer.shape.as_list()[-1]), [3, 3], stride=1)
    conv_input = layers.conv2d(
        conv_input, max(4, layer.shape.as_list()[-1]), [3, 3], stride=1)
    conv_output2 = layers.conv2d(
        conv_input, max(4, layer.shape.as_list()[-1]), [3, 3], stride=1)
    conv_output = tf.concat([conv_output, conv_output2], axis=-1)

    return upsampled_motion_field + layers.conv2d(
        conv_output,
        motion_field.shape.as_list()[-1], [1, 1],
        stride=1,
        activation_fn=None,
        biases_initializer=None,
        scope=layer.op.name + '/MotionBottleneck')


def tensorflow_demo2():
    motion_field = tf.placeholder(tf.float32, shape=[1, 3])
    layer = tf.placeholder(tf.float32, shape=[1, 1, 4, 1024])

    y = _refine_motion_field(motion_field, layer)

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    motion_vars = np.random.rand(1, 3)
    layer_vars = np.random.rand(1, 1, 4, 1024)

    sess.run(y, feed_dict={motion_field: motion_vars, layer: layer_vars})
    print(y.shape)
    pass


def tensorflow_demo3():
    motion_field = tf.placeholder(tf.float32, shape=[1, 2, 7, 3])
    layer = tf.placeholder(tf.float32, shape=[1, 4, 13, 256])

    _, h, w, _ = tf.unstack(tf.shape(layer))
    upsampled_motion_field = tf.image.resize_bilinear(motion_field, [h, w])

    y = upsampled_motion_field

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    motion_vars = np.random.rand(1, 2, 7, 3)
    layer_vars = np.random.rand(1, 4, 13, 256)

    sess.run(y, feed_dict={motion_field: motion_vars, layer: layer_vars})
    print(y.shape)
    pass

def add_intrinsics_head(bottleneck, image_height, image_width):
    with tf.variable_scope('CameraIntrinsics'):

        focal_lengths = tf.squeeze(
            layers.conv2d(bottleneck, 2, [1, 1], stride=1, activation_fn=tf.nn.softplus, weights_regularizer=None, scope='foci'),
            axis=(1, 2)
        ) * tf.to_float(tf.convert_to_tensor([[image_width, image_height]]))

        offsets = (tf.squeeze(
            layers.conv2d(bottleneck, 2, [1, 1], stride=1, activation_fn=None, weights_regularizer=None, biases_initializer=None, scope='offsets'),
            axis=(1, 2)) + 0.5
        ) * tf.to_float(tf.convert_to_tensor([[image_width, image_height]]))

        foci = tf.linalg.diag(focal_lengths)

        intrinsic_mat = tf.concat([foci, tf.expand_dims(offsets, -1)], axis=2)
        batch_size = tf.shape(bottleneck)[0]
        last_row = tf.tile([[[0.0, 0.0, 1.0]]], [batch_size, 1, 1])
        intrinsic_mat = tf.concat([intrinsic_mat, last_row], axis=1)
        return intrinsic_mat

def tensorflow_demox():
    bottleneck = tf.placeholder(tf.float32, shape=[1, 1, 1, 1024])
    image_height = 128
    image_width = 416
    pass

if __name__ == '__main__':
    tensorflow_demo()
