import torch
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers
import random


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
    y = layers.conv2d(
        y,
        128, [3, 3],
        stride=2,
        scope='Conv4',
        weights_regularizer=layers.l2_regularizer(weight_reg),
        activation_fn=tf.nn.relu)
    y = layers.conv2d(
        y,
        256, [3, 3],
        stride=2,
        scope='Conv5',
        weights_regularizer=layers.l2_regularizer(weight_reg),
        activation_fn=tf.nn.relu)
    y = layers.conv2d(
        y,
        512, [3, 3],
        stride=2,
        scope='Conv6',
        weights_regularizer=layers.l2_regularizer(weight_reg),
        activation_fn=tf.nn.relu)
    y = layers.conv2d(
        y,
        1024, [3, 3],
        stride=2,
        scope='Conv7',
        weights_regularizer=layers.l2_regularizer(weight_reg),
        activation_fn=tf.nn.relu)
    y = tf.reduce_mean(y, axis=[1, 2], keepdims=True)
    y = layers.conv2d(
        y,
        6, [1, 1],
        stride=1,
        activation_fn=None,
        biases_initializer=None,
        scope='background_motion')

    y = y[:, :, :, 3:]

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
    motion_field = tf.placeholder(tf.float32, shape=[1, 64, 208, 3])
    layer = tf.placeholder(tf.float32, shape=[1, 128, 416, 6])

    y = _refine_motion_field(motion_field, layer)

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    motion_vars = np.random.rand(1, 64, 208, 3)
    layer_vars = np.random.rand(1, 128, 416, 6)

    sess.run(y, feed_dict={motion_field: motion_vars, layer: layer_vars})
    print(y.shape)
    pass


def tensorflow_demo3():
    motion_field = tf.placeholder(tf.float32, shape=[4, 1, 1, 3])
    layer = tf.placeholder(tf.float32, shape=[4, 1, 4, 1024])

    _, h, w, _ = tf.unstack(tf.shape(layer))
    upsampled_motion_field = tf.image.resize_bilinear(motion_field, [h, w])
    conv_input = tf.concat([upsampled_motion_field, layer], axis=3)
    conv_output = layers.conv2d(conv_input, max(4, layer.shape.as_list()[-1]), [3, 3], stride=1)
    conv_input2 = layers.conv2d(conv_input, max(4, layer.shape.as_list()[-1]), [3, 3], stride=1)
    conv_output2 = layers.conv2d(conv_input2, max(4, layer.shape.as_list()[-1]), [3, 3], stride=1)
    conv_output3 = tf.concat([conv_output, conv_output2], axis=-1)

    conv_output4 = layers.conv2d(conv_output3, motion_field.shape.as_list()[-1], [1, 1], stride=1,
                                 activation_fn=None, biases_initializer=None, scope=layer.op.name + '/MotionBottleneck')
    out = conv_output4 + upsampled_motion_field

    y = conv_output

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    motion_vars = np.random.rand(4, 1, 1, 3)
    layer_vars = np.random.rand(4, 1, 4, 1024)

    sess.run(y, feed_dict={motion_field: motion_vars, layer: layer_vars})
    print(y.shape)
    pass


def create_scales(constraint_minimum):
    """Creates variables representing rotation and translation scaling factors.

  Args:
    constraint_minimum: A scalar, the variables will be constrained to not fall
      below it.

  Returns:
    Two scalar variables, rotation and translation scale.
  """

    def constraint(x):
        return tf.nn.relu(x - constraint_minimum) + constraint_minimum

    with tf.variable_scope('Scales', initializer=0.01, constraint=constraint):
        rot_scale = tf.get_variable('rotation')
        trans_scale = tf.get_variable('translation')

    return rot_scale, trans_scale


def add_intrinsics_head(bottleneck, image_height, image_width):
    with tf.variable_scope('CameraIntrinsics'):
        focal_lengths = tf.squeeze(
            layers.conv2d(bottleneck, 2, [1, 1], stride=1, activation_fn=tf.nn.softplus, weights_regularizer=None,
                          scope='foci'),
            axis=(1, 2)
        ) * tf.to_float(tf.convert_to_tensor([[image_width, image_height]]))

        offsets = (tf.squeeze(
            layers.conv2d(bottleneck, 2, [1, 1], stride=1, activation_fn=None, weights_regularizer=None,
                          biases_initializer=None, scope='offsets'),
            axis=(1, 2)
        ) + 0.5) * tf.to_float(tf.convert_to_tensor([[image_width, image_height]]))

        foci = tf.linalg.diag(focal_lengths)

        intrinsic_mat = tf.concat([foci, tf.expand_dims(offsets, -1)], axis=2)
        batch_size = tf.shape(bottleneck)[0]
        last_row = tf.tile([[[0.0, 0.0, 1.0]]], [batch_size, 1, 1])
        intrinsic_mat = tf.concat([intrinsic_mat, last_row], axis=1)
        return intrinsic_mat


def tensorflow_demo4():
    images = tf.placeholder(tf.float32, shape=[1, 128, 416, 6])
    bottleneck = tf.placeholder(tf.float32, shape=[1, 1, 1, 1024])

    image_height, image_width = tf.unstack(tf.shape(images)[1:3])

    y = add_intrinsics_head(bottleneck, image_height, image_width)

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    images_vars = np.random.rand(1, 128, 416, 6)
    bottleneck_vars = np.random.rand(1, 1, 1, 1024)

    sess.run(y, feed_dict={images: images_vars, bottleneck: bottleneck_vars})
    print(y.shape)

    pass


WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'


def s_conv(x, filter_size, out_channel, stride, pad='SAME', input_q=None, output_q=None, name='conv'):
    """Helper function for defining ResNet architecture."""
    if (input_q is None) ^ (output_q is None):
        raise ValueError('Input/Output splits are not correctly given.')

    in_shape = x.get_shape()
    with tf.variable_scope(name):
        # Main operation: conv2d
        with tf.device('/CPU:0'):
            kernel = tf.get_variable('kernel',
                                     [filter_size, filter_size, in_shape[3], out_channel],
                                     tf.float32,
                                     initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(2.0 / filter_size / filter_size / out_channel)
                                     )
                                     )
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
        conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], pad)
    return conv


def s_relu(x, name=None, leakness=0.0):
    """Helper function for defining ResNet architecture."""
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x * leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')


def _bn(x, is_train, name='bn'):
    """Helper function for defining ResNet architecture."""
    bn = tf.layers.batch_normalization(x, training=is_train, name=name)
    return bn


def s_residual_block(x,
                     is_training,
                     input_q=None,
                     output_q=None,
                     name='unit',
                     normalizer_fn=None):
    """Helper function for defining ResNet architecture."""
    normalizer_fn = normalizer_fn or _bn
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        shortcut = x  # Shortcut connection
        # Residual
        x = s_conv(
            x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
        x = normalizer_fn(x, is_train=is_training, name='bn_1')
        x = s_relu(x, name='relu_1')
        x = s_conv(
            x,
            3,
            num_channel,
            1,
            input_q=output_q,
            output_q=output_q,
            name='conv_2')
        x = normalizer_fn(x, is_train=is_training, name='bn_2')
        # Merge
        x = x + shortcut
        x = s_relu(x, name='relu_2')
    return x


def s_residual_block_first(x,
                           is_training,
                           out_channel,
                           strides,
                           name='unit',
                           normalizer_fn=None):
    """Helper function for defining ResNet architecture."""
    normalizer_fn = normalizer_fn or _bn
    in_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        # Shortcut connection
        if in_channel == out_channel:
            if strides == 1:
                shortcut = tf.identity(x)
            else:
                shortcut = tf.nn.max_pool(x, [1, strides, strides, 1],
                                          [1, strides, strides, 1], 'VALID')
        else:
            shortcut = s_conv(x, 1, out_channel, strides, name='shortcut')
        # Residual
        x = s_conv(x, 3, out_channel, strides, name='conv_1')
        x = normalizer_fn(x, is_train=is_training, name='bn_1')
        x = s_relu(x, name='relu_1')
        x = s_conv(x, 3, out_channel, 1, name='conv_2')
        x = normalizer_fn(x, is_train=is_training, name='bn_2')
        # Merge
        x = x + shortcut
        x = s_relu(x, name='relu_2')
    return x


def _concat_and_pad(decoder_layer, encoder_layer, padding_mode):
    concat = tf.concat([decoder_layer, encoder_layer], axis=3)
    return tf.pad(concat, [[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_mode)


def depthnet_demo():
    target_image = tf.placeholder(tf.float32, shape=[1, 128, 416, 3])
    econv1 = tf.placeholder(tf.float32, shape=[1, 64, 208, 64])
    econv2 = tf.placeholder(tf.float32, shape=[1, 32, 104, 64])
    econv3 = tf.placeholder(tf.float32, shape=[1, 16, 52, 128])
    econv4 = tf.placeholder(tf.float32, shape=[1, 8, 26, 256])
    econv5 = tf.placeholder(tf.float32, shape=[1, 4, 13, 512])

    decoder_filters = [16, 32, 64, 128, 256]
    bottleneck = econv5
    reg = layers.l2_regularizer(0.1)
    padding_mode = 'CONSTANT'  # 'REFLECT'
    with contrib_framework.arg_scope([layers.conv2d, layers.conv2d_transpose],
                                     normalizer_fn=None,
                                     normalizer_params=None,
                                     activation_fn=tf.nn.relu,
                                     weights_regularizer=reg):
        upconv5 = layers.conv2d_transpose(bottleneck, decoder_filters[4], [3, 3], stride=2,
                                          scope='upconv5')  # (1, 8, 26, 256)
        x5 = _concat_and_pad(upconv5, econv4, padding_mode)  # (1, 10, 28, 512)
        iconv5 = layers.conv2d(x5, decoder_filters[4], [3, 3], stride=1, scope='iconv5',
                               padding='VALID')  # (1, 8, 26, 256)

        upconv4 = layers.conv2d_transpose(iconv5, decoder_filters[3], [3, 3], stride=2,
                                          scope='upconv4')  # (1, 16, 52, 128)
        x4 = _concat_and_pad(upconv4, econv3, padding_mode)  # (1, 18, 54, 256)
        iconv4 = layers.conv2d(x4, decoder_filters[3], [3, 3], stride=1, scope='iconv4',
                               padding='VALID')  # (1, 16, 52, 128)

        upconv3 = layers.conv2d_transpose(iconv4, decoder_filters[2], [3, 3], stride=2,
                                          scope='upconv3')  # (1, 32, 104, 64)
        x3 = _concat_and_pad(upconv3, econv2, padding_mode)  # (1, 34, 106, 128)
        iconv3 = layers.conv2d(x3, decoder_filters[2], [3, 3], stride=1, scope='iconv3',
                               padding='VALID')  # (1, 32, 104, 64)

        upconv2 = layers.conv2d_transpose(iconv3, decoder_filters[1], [3, 3], stride=2,
                                          scope='upconv2')  # (1, 64, 208, 32)
        x2 = _concat_and_pad(upconv2, econv1, padding_mode)  # (1, 66, 210, 96)
        iconv2 = layers.conv2d(x2, decoder_filters[1], [3, 3], stride=1, scope='iconv2',
                               padding='VALID')  # (1, 64, 208, 32)

        upconv1 = layers.conv2d_transpose(iconv2, decoder_filters[0], [3, 3], stride=2,
                                          scope='upconv1')  # (1, 128, 416, 16)
        x1 = tf.pad(upconv1, [[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_mode)  # (1, 130, 418, 16)
        iconv1 = layers.conv2d(x1, decoder_filters[0], [3, 3], stride=1, scope='iconv1',
                               padding='VALID')  # (1, 128, 416, 16)

        depth_input = tf.pad(iconv1, [[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_mode)  # (1, 130, 418, 16)

        out = layers.conv2d(depth_input, 1, [3, 3], stride=1, activation_fn=tf.nn.softplus, normalizer_fn=None,
                            scope='disp1', padding='VALID')  # (1, 128, 416, 1)

    y = iconv5

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    target_image_vars = np.random.rand(1, 128, 416, 3)
    econv1_vars = np.random.rand(1, 64, 208, 64)
    econv2_vars = np.random.rand(1, 32, 104, 64)
    econv3_vars = np.random.rand(1, 16, 52, 128)
    econv4_vars = np.random.rand(1, 8, 26, 256)
    econv5_vars = np.random.rand(1, 4, 13, 512)
    sess.run(y, feed_dict={target_image: target_image_vars,
                           econv1: econv1_vars,
                           econv2: econv2_vars,
                           econv3: econv3_vars,
                           econv4: econv4_vars,
                           econv5: econv5_vars})
    print(" input shappe: ", end='')
    print(target_image.shape)
    print("output shappe: ", end='')
    print(y.shape)
    pass


def create_scales(constraint_minimum):
    def constraint(x):
        return tf.nn.relu(x - constraint_minimum) + constraint_minimum

    with tf.variable_scope('Scales', initializer=0.01, constraint=constraint):
        rot_scale = tf.get_variable('rotation')
        trans_scale = tf.get_variable('translation')

    return rot_scale, trans_scale


def scale_demo():
    r, t = create_scales(0.001)
    r = r-t
    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)
    # Evaluate the tensor `c`.
    print(sess.run(r))

    pass


if __name__ == '__main__':
    tensorflow_demo3()
