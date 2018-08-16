# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
from .tcn import tcn

layers_dict = {
    "conv1d": tf.layers.conv1d,
    "conv2d": tf.layers.conv2d,
    "tcn": tcn,
}


def conv_actv(layer_type, name, inputs, filters, kernel_size, activation_fn, strides,
              padding, regularizer, training, data_format, use_residual):
  """Helper function that applies convolution and activation.
    Args:
      layer_type: the following types are supported
        'conv1d', 'conv2d'
  """
  layer = layers_dict[layer_type]

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


def conv_bn_actv(layer_type, name, inputs, filters, kernel_size, activation_fn, strides,
                 padding, regularizer, training, data_format, use_residual, bn_momentum,
                 bn_epsilon):
  """Helper function that applies convolution, batch norm and activation.
    Accepts inputs in 'channels_last' format only.
    Args:
      layer_type: the following types are supported
        'conv1d', 'conv2d'
  """
  layer = layers_dict[layer_type]

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
  if layer_type == "conv1d":
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

  # add residual connections
  if use_residual:
    in_size_index = -1 if data_format == 'channels_last' else 1
    channels = int(inputs.get_shape()[in_size_index])
    if channels != filters:
      residual = tf.layers.dense(
          inputs=inputs,
          units=filters,
          kernel_regularizer=regularizer,
          name="{}/residual".format(name),
      )
    else:
      residual = inputs
    bn = bn + residual

  output = bn
  if activation_fn is not None:
    output = activation_fn(output)
  return output
