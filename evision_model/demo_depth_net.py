import torch
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers
import random


def normalize(x, is_train, name='bn', stddev=0.5):
    """Applies layer normalization and applies noise on the mean and variance.

  Args:
    x: tf.Tensor to normalize, of shape [B, H, W, C].
    is_train: A boolean, True at training mode.
    name: A string, a name scope.
    stddev: Standard deviation of the Gaussian noise. Defaults to 0.5 because
      this is the largest value where the noise is guaranteed to be a
      non-negative multiplicative factor

  Returns:
    A tf.Tensor of shape [B, H, W, C], the normalized tensor.
  """

    with tf.variable_scope(name, None, [x]):
        inputs_shape = x.shape.as_list()
        params_shape = inputs_shape[-1:]
        beta = tf.get_variable(
            'beta', shape=params_shape, initializer=tf.initializers.zeros())
        gamma = tf.get_variable(
            'gamma', shape=params_shape, initializer=tf.initializers.ones())
        mean, variance = tf.nn.moments(x, [1, 2], keep_dims=True)
        if is_train:
            mean *= 1.0 + tf.random.truncated_normal(tf.shape(mean), stddev=stddev)
            variance *= 1.0 + tf.random.truncated_normal(
                tf.shape(variance), stddev=stddev)
        outputs = tf.nn.batch_normalization(
            x,
            mean,
            variance,
            offset=beta,
            scale=gamma,
            variance_epsilon=1e-3)
        outputs.set_shape(x.shape)
    return outputs


def demon_normalize():
    x = tf.placeholder(tf.float32, shape=[5, 128, 416, 3])
    stddev = 0.5
    with tf.variable_scope('nor', None, [x]):
        inputs_shape = x.shape.as_list()
        params_shape = inputs_shape[-1:]
        beta = tf.get_variable('beta', shape=params_shape, initializer=tf.initializers.zeros())
        gamma = tf.get_variable('gamma', shape=params_shape, initializer=tf.initializers.ones())
        mean, variance = tf.nn.moments(x, [1, 2], keep_dims=True)

        mean *= 1.0 + tf.random.truncated_normal(tf.shape(mean), stddev=stddev)
        variance *= 1.0 + tf.random.truncated_normal(tf.shape(variance), stddev=stddev)
        outputs = tf.nn.batch_normalization(x, mean, variance, offset=beta, scale=gamma, variance_epsilon=1e-3)
        outputs.set_shape(x.shape)

    y = mean

    initialize_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initialize_op)

    target_image_vars = np.random.rand(5, 128, 416, 3)

    sess.run(y, feed_dict={x: target_image_vars})
    print(" input shappe: ", end='')
    print(x.shape)
    print("output shappe: ", end='')
    print(y.shape)
    pass


if __name__ == '__main__':
    demon_normalize()
    pass