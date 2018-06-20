"""Implementation of fully connected network with weight normalization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class FeedFowardNetworkNormalized(tf.layers.Layer):
  """Fully connected feedforward network with weight normalization"""
  """Inspired from https://github.com/tobyyouup/conv_seq2seq"""

  def __init__(self, in_dim, out_dim, dropout, var_scope_name):
    super(FeedFowardNetworkNormalized, self).__init__()
    self.out_dim = out_dim
    self.in_dim = in_dim

    with tf.variable_scope(var_scope_name):
      # use weight normalization (Salimans & Kingma, 2016)  w = g * v/2-norm(v)
      self.V = tf.get_variable('V', shape=[int(in_dim), out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                            dropout * 1.0 / int(in_dim))), trainable=True)
      self.V_norm = tf.norm(self.V.initialized_value(), axis=0)
      self.g = tf.get_variable('g', dtype=tf.float32, initializer=self.V_norm, trainable=True)
      self.b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32,
                               initializer=tf.zeros_initializer(), trainable=True)

  def call(self, x):

    batch_size = tf.shape(x)[0]

    x = tf.reshape(x, [-1, self.in_dim])
    output = tf.matmul(x, self.V)
    output = tf.reshape(output, [batch_size, -1, self.out_dim])

    # x*(v*(g/2-norm(v))) + b
    scaler = tf.div(self.g, tf.norm(self.V, axis=0))
    output = tf.reshape(scaler, [1, self.out_dim]) * output + tf.reshape(self.b, [1, self.out_dim])

    return output

