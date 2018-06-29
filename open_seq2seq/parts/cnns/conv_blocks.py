# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf


def conv_actv(type, name, inputs, filters, kernel_size, activation_fn, strides,
              padding, regularizer, training, data_format):
  """Helper function that applies convolution and activation.
    Args:
      type: the following types are supported
        'conv1d', 'conv2d'
  """
  if type == "conv1d":
    layer = tf.layers.conv1d
  elif type == "conv2d":
    layer = tf.layers.conv2d

  conv = layer(
      name="{}".format(name),
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      kernel_regularizer=regularizer,
      use_bias=False,
      data_format=data_format,
  )

  output = conv
  if activation_fn is not None:
    output = activation_fn(output)
  return output


def conv_bn_actv(type, name, inputs, filters, kernel_size, activation_fn, strides,
                 padding, regularizer, training, data_format, bn_momentum,
                 bn_epsilon):
  """Helper function that applies convolution, batch norm and activation.
    Accepts inputs in 'channels_last' format only.
    Args:
      type: the following types are supported
        'conv1d', 'conv2d'
  """
  if type == "conv1d":
    layer = tf.layers.conv1d
  elif type == "conv2d":
    layer = tf.layers.conv2d

  conv = layer(
      name="{}".format(name),
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      kernel_regularizer=regularizer,
      use_bias=False,
      data_format=data_format,
  )

  # trick to make batchnorm work for mixed precision training.
  # To-Do check if batchnorm works smoothly for >4 dimensional tensors
  squeeze = False
  if type == "conv1d":
    conv = tf.expand_dims(conv, axis=1)  # NWC --> NHWC
    squeeze = True

  bn = tf.layers.batch_normalization(
      name="{}/bn".format(name),
      inputs=conv,
      gamma_regularizer=regularizer,
      training=training,
      axis=-1 if data_format == 'channels_last' else 1,
      momentum=bn_momentum,
      epsilon=bn_epsilon,
  )

  if squeeze:
    bn = tf.squeeze(bn, axis=1)

  output = bn
  if activation_fn is not None:
    output = activation_fn(output)
  return output
