# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf


class TemporalConvolutionalLayer(tf.layers.Conv1D):
  """Temporal Convolutional layer
  """

  def __init__(
      self,
      filters,
      kernel_size,
      strides=1,
      dilation_rate=1,
      activation=None,
      data_format='channels_last',
      name="temporal_convolutional",
      use_bias=True,
      kernel_initializer=None,
      bias_initializer=tf.zeros_initializer(),
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      trainable=True,
      padding='valid',
      **kwargs
  ):
    super(TemporalConvolutionalLayer, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        data_format=data_format,
        name=name,
        padding='valid',
        **kwargs
    )

  def call(self, inputs):
    pads = (self.kernel_size[0] - 1) * self.dilation_rate[0]
    padding = tf.fill([tf.shape(inputs)[0], pads, tf.shape(
        inputs)[2]], tf.constant(0, dtype=inputs.dtype))
    inputs = tf.concat([padding, inputs], 1)
    return super(TemporalConvolutionalLayer, self).call(inputs)


def tcn(inputs,
        filters,
        kernel_size,
        strides=1,
        padding='valid',
        data_format='channels_last',
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None):
  """Functional interface for temporal convolution layer.
  """
  layer = TemporalConvolutionalLayer(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      trainable=trainable,
      name=name,
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs)
