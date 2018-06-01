# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
from .resnet_blocks import conv, pool
from .encoder import Encoder


class AlexNetEncoder(Encoder):
  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
    })

  def __init__(self, params, model, name="resnet_encoder", mode='train'):
    super(AlexNetEncoder, self).__init__(params, model, name, mode)

  def _encode(self, input_dict):
    regularizer = self.params.get('regularizer', None)
    data_format = self.params.get('data_format', 'channels_first')
    if self.mode == 'train':
      dropout_keep_prob = self.params.get('dropout_keep_prob', 0.5)
    else:
      dropout_keep_prob = 1.0

    x = input_dict['source_tensors'][0]

    if data_format == 'channels_first':
      x = tf.transpose(x, [0, 3, 1, 2])

    x = conv(x, filters=64, kernel_size=(11, 11), strides=(4, 4),
             data_format=data_format, padding='VALID', regularizer=regularizer)
    x = pool(x, pool_size=(3, 3), data_format=data_format)
    x = conv(x, filters=192, kernel_size=(5, 5),
             data_format=data_format, regularizer=regularizer)
    x = pool(x, pool_size=(3, 3), data_format=data_format)
    x = conv(x, filters=384, kernel_size=(3, 3),
             data_format=data_format, regularizer=regularizer)
    x = conv(x, filters=256, kernel_size=(3, 3),
             data_format=data_format, regularizer=regularizer)
    x = conv(x, filters=256, kernel_size=(3, 3),
             data_format=data_format, regularizer=regularizer)
    x = pool(x, pool_size=(3, 3), data_format=data_format)

    if data_format == 'channels_first':
      x = tf.transpose(x, [0, 2, 3, 1])
    input_shape = x.get_shape().as_list()
    num_inputs = input_shape[1] * input_shape[2] * input_shape[3]
    x = tf.reshape(x, [-1, num_inputs])

    x = tf.layers.dense(x, 4096, activation=tf.nn.relu,
                        kernel_regularizer=regularizer)
    x = tf.nn.dropout(x=x, keep_prob=dropout_keep_prob)
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu,
                        kernel_regularizer=regularizer)
    x = tf.nn.dropout(x=x, keep_prob=dropout_keep_prob)

    return {'outputs': x}
