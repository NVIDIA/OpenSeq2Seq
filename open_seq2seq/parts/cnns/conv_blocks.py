# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf

layers_dict = {
    "conv1d": tf.layers.conv1d,
    "conv2d": tf.layers.conv2d
}

def conv_actv(layer_type, name, inputs, filters, kernel_size, activation_fn, strides,
              padding, regularizer, training, data_format, dilation=1):
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
      dilation_rate=dilation,
      kernel_regularizer=regularizer,
      use_bias=False,
      data_format=data_format,
  )

  output = conv
  if activation_fn is not None:
    output = activation_fn(output)
  return output


def conv_bn_actv(layer_type, name, inputs, filters, kernel_size, activation_fn, strides,
                 padding, regularizer, training, data_format, bn_momentum,
                 bn_epsilon, dilation=1):
  """Helper function that applies convolution, batch norm and activation.
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
      dilation_rate=dilation,
      kernel_regularizer=regularizer,
      use_bias=False,
      data_format=data_format,
  )

  # trick to make batchnorm work for mixed precision training.
  # To-Do check if batchnorm works smoothly for >4 dimensional tensors
  squeeze = False
  if layer_type == "conv1d":
    axis = 1 if data_format == 'channels_last' else 2
    conv = tf.expand_dims(conv, axis=axis)  # NWC --> NHWC
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
    bn = tf.squeeze(bn, axis=axis)

  output = bn
  if activation_fn is not None:
    output = activation_fn(output)
  return output


def conv_ln_actv(layer_type, name, inputs, filters, kernel_size, activation_fn, strides,
                 padding, regularizer, training, data_format, dilation=1):
  """Helper function that applies convolution, layer norm and activation.
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
      dilation_rate=dilation,
      kernel_regularizer=regularizer,
      use_bias=False,
      data_format=data_format,
  )

  if data_format == 'channels_first':
    if layer_type == "conv1d":
      conv = tf.transpose(conv, [0, 2, 1])
    elif layer_type == "conv2d":
      conv = tf.transpose(conv, [0, 2, 3, 1])
  ln = tf.contrib.layers.layer_norm(
      inputs=conv,
  )
  if data_format == 'channels_first':
    if layer_type == "conv1d":
      ln = tf.transpose(ln, [0, 2, 1])
    elif layer_type == "conv2d":
      ln = tf.transpose(ln, [0, 3, 1, 2])

  output = ln
  if activation_fn is not None:
    output = activation_fn(output)
  return output

def conv_in_actv(layer_type, name, inputs, filters, kernel_size, activation_fn, strides,
                 padding, regularizer, training, data_format, dilation=1):
  """Helper function that applies convolution, instance norm and activation.
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
      dilation_rate=dilation,
      kernel_regularizer=regularizer,
      use_bias=False,
      data_format=data_format,
  )

  sn = tf.contrib.layers.instance_norm(
      inputs=conv,
      data_format="NHWC" if data_format == 'channels_last' else "NCHW"
  )

  output = sn
  if activation_fn is not None:
    output = activation_fn(output)
  return output

