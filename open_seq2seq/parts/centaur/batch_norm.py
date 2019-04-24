# Copyright (c) 2019 NVIDIA Corporation
import tensorflow as tf


class BatchNorm1D:
  """
  1D batch normalization layer.
  """

  def __init__(self, *args, **kwargs):
    super(BatchNorm1D, self).__init__()
    self.norm = tf.layers.BatchNormalization(*args, **kwargs)

  def __call__(self, x, training):
    with tf.variable_scope("batch_norm_1d"):
      y = tf.expand_dims(x, axis=1)
      y = self.norm(y, training=training)
      y = tf.squeeze(y, axis=1)
      return y
