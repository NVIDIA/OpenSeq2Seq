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
      'resnet_size': int,
      'block_sizes': list,
      'block_strides': list,
      'version': [1, 2],
      'bottleneck': bool,
      'final_size': int,
      'first_num_filters': int,
      'first_kernel_size': int,
      'first_conv_stride': int,
      'first_pool_size': int,
      'first_pool_stride': int,
      'data_format': ['channels_first', 'channels_last'],
    })

  def __init__(self, params, model, name="resnet_encoder", mode='train'):
    super(ResNetEncoder, self).__init__(params, model, name, mode)

  def _encode(self, input_dict):
    inputs = input_dict['source_tensors'][0]
    if 'resnet_size' not in self.params and 'block_sizes' not in self.params:
      raise ValueError('Either "resnet_size" or "block_sizes" '
                       'have to be specified in the config')
    if 'resnet_size' in self.params and 'block_sizes' in self.params:
      raise ValueError('"resnet_size" and "block_sizes" cannot '
                       'be specified together')
    if 'resnet_size' in self.params:
      if self.params['resnet_size'] < 50:
        bottleneck = self.params.get('bottleneck', False)
        final_size = self.params.get('final_size', 512)
      else:
        bottleneck = self.params.get('bottleneck', True)
        final_size = self.params.get('final_size', 2048)
      block_sizes_dict = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3],
      }
      block_sizes = block_sizes_dict[self.params['resnet_size']]
    else:
      if 'bottleneck' not in self.params:
        raise ValueError('If "resnet_size" not specified you have to provide '
                         '"bottleneck" parameter')
      if 'final_size' not in self.params:
        raise ValueError('If "resnet_size" not specified you have to provide '
                         '"final_size" parameter')
      bottleneck = self.params['bottleneck']
      final_size = self.params['final_size']
      block_sizes = self.params['block_sizes']

    num_filters = self.params.get('first_num_filters', 64)
    kernel_size = self.params.get('first_kernel_size', 7)
    conv_stride = self.params.get('first_conv_stride', 2)
    first_pool_size = self.params.get('first_pool_size', 3)
    first_pool_stride = self.params.get('first_pool_stride', 2)

    block_strides = self.params.get('block_strides', [1, 2, 2, 2])
    version = self.params.get('version', 2)
    data_format = self.params.get('data_format', 'channels_first')

    if bottleneck:
      if version == 1:
        block_fn = bottleneck_block_v1
      else:
        block_fn = bottleneck_block_v2
    else:
      if version == 1:
        block_fn = building_block_v1
      else:
        block_fn = building_block_v2

    training = self.mode == 'train'
    regularizer = self.params.get('regularizer', None)

    if data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=num_filters, kernel_size=kernel_size,
      strides=conv_stride, data_format=data_format, regularizer=regularizer,
    )
    inputs = tf.identity(inputs, 'initial_conv')

    if first_pool_size:
      inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=first_pool_size,
        strides=first_pool_stride, padding='SAME',
        data_format=data_format,
      )
      inputs = tf.identity(inputs, 'initial_max_pool')

    for i, num_blocks in enumerate(block_sizes):
      num_filters = num_filters * (2**i)
      inputs = block_layer(
        inputs=inputs, filters=num_filters, bottleneck=bottleneck,
        block_fn=block_fn, blocks=num_blocks,
        strides=block_strides[i], training=training,
        name='block_layer{}'.format(i + 1), data_format=data_format,
        regularizer=regularizer,
      )

    inputs = batch_norm(inputs, training, data_format, regularizer=regularizer)
    inputs = tf.nn.relu(inputs)

    # The current top layer has shape
    # `batch_size x pool_size x pool_size x final_size`.
    # ResNet does an Average Pooling layer over pool_size,
    # but that is the same as doing a reduce_mean. We do a reduce_mean
    # here because it performs better than AveragePooling2D.
    axes = [2, 3] if data_format == 'channels_first' else [1, 2]
    inputs = tf.reduce_mean(inputs, axes, keepdims=True)
    inputs = tf.identity(inputs, 'final_reduce_mean')

    outputs = tf.reshape(inputs, [-1, final_size])

    return {'outputs': outputs}
