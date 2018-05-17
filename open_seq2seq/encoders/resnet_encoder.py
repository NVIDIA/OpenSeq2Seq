# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
from .resnet_blocks import conv2d_fixed_padding, batch_norm, block_layer, \
                           bottleneck_block_v1, bottleneck_block_v2, \
                           building_block_v1, building_block_v2
from .encoder import Encoder


class ResNetEncoder(Encoder):
  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{

    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{

    })

  def __init__(self, params, model, name="resnet_encoder", mode='train'):
    super(ResNetEncoder, self).__init__(params, model, name, mode)

  def _encode(self, input_dict):
    inputs = input_dict['source_tensors'][0]

    self.resnet_size = 50
    if self.resnet_size < 50:
      self.bottleneck = False
      self.final_size = 512
    else:
      self.bottleneck = True
      self.final_size = 2048

    self.num_filters = 64
    self.kernel_size = 7
    self.conv_stride = 2
    self.first_pool_size = 3
    self.first_pool_stride = 2

    block_sizes = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3],
    }

    self.block_sizes = block_sizes[self.resnet_size]
    self.block_strides = [1, 2, 2, 2]
    self.version = 2
    self.data_format = 'channels_first'
    self.num_classes = 1001

    if self.bottleneck:
      if self.version == 1:
        self.block_fn = bottleneck_block_v1
      else:
        self.block_fn = bottleneck_block_v2
    else:
      if self.version == 1:
        self.block_fn = building_block_v1
      else:
        self.block_fn = building_block_v2

    training = self.mode == 'train'
    regularizer = self.params.get('regularizer', None)

    if self.data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
      strides=self.conv_stride, data_format=self.data_format,
      regularizer=regularizer,
    )
    inputs = tf.identity(inputs, 'initial_conv')

    if self.first_pool_size:
      inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=self.first_pool_size,
        strides=self.first_pool_stride, padding='SAME',
        data_format=self.data_format,
      )
      inputs = tf.identity(inputs, 'initial_max_pool')

    for i, num_blocks in enumerate(self.block_sizes):
      num_filters = self.num_filters * (2**i)
      inputs = block_layer(
        inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
        block_fn=self.block_fn, blocks=num_blocks,
        strides=self.block_strides[i], training=training,
        name='block_layer{}'.format(i + 1), data_format=self.data_format,
        regularizer=regularizer,
      )

    inputs = batch_norm(inputs, training, self.data_format,
                        regularizer=regularizer)
    inputs = tf.nn.relu(inputs)

    # The current top layer has shape
    # `batch_size x pool_size x pool_size x final_size`.
    # ResNet does an Average Pooling layer over pool_size,
    # but that is the same as doing a reduce_mean. We do a reduce_mean
    # here because it performs better than AveragePooling2D.
    axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
    inputs = tf.reduce_mean(inputs, axes, keepdims=True)
    inputs = tf.identity(inputs, 'final_reduce_mean')

    outputs = tf.reshape(inputs, [-1, self.final_size])

    return {'outputs': outputs}
