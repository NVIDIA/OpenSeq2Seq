# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
import numpy as np

from .model import Model
from open_seq2seq.utils.utils import deco_print
from .resnet_model import conv2d_fixed_padding, batch_norm, block_layer, \
                          _bottleneck_block_v1, _bottleneck_block_v2, \
                          _building_block_v1, _building_block_v2


class ResNet(Model):
  @staticmethod
  def get_optional_params():
    return dict(Model.get_optional_params(), **{
      'weight_decay': float,
      'regularize_bn': bool,
      'resnet_size': int,
    })

  def _build_forward_pass_graph(self, input_tensors, gpu_id=0):
    inputs, labels = input_tensors

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
        self.block_fn = _bottleneck_block_v1
      else:
        self.block_fn = _bottleneck_block_v2
    else:
      if self.version == 1:
        self.block_fn = _building_block_v1
      else:
        self.block_fn = _building_block_v2

    training = self.mode == 'train'

    with tf.variable_scope("ForwardPass"):
      if self.data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

      inputs = conv2d_fixed_padding(
          inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
          strides=self.conv_stride, data_format=self.data_format)
      inputs = tf.identity(inputs, 'initial_conv')

      if self.first_pool_size:
        inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=self.first_pool_size,
            strides=self.first_pool_stride, padding='SAME',
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')

      for i, num_blocks in enumerate(self.block_sizes):
        num_filters = self.num_filters * (2**i)
        inputs = block_layer(
            inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
            block_fn=self.block_fn, blocks=num_blocks,
            strides=self.block_strides[i], training=training,
            name='block_layer{}'.format(i + 1), data_format=self.data_format)

      inputs = batch_norm(inputs, training, self.data_format)
      inputs = tf.nn.relu(inputs)

      # The current top layer has shape
      # `batch_size x pool_size x pool_size x final_size`.
      # ResNet does an Average Pooling layer over pool_size,
      # but that is the same as doing a reduce_mean. We do a reduce_mean
      # here because it performs better than AveragePooling2D.
      axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
      inputs = tf.reduce_mean(inputs, axes, keepdims=True)
      inputs = tf.identity(inputs, 'final_reduce_mean')

      inputs = tf.reshape(inputs, [-1, self.final_size])
      inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
      logits = tf.identity(inputs, 'final_dense')

      if self.mode == "train" or self.mode == "eval":
        with tf.variable_scope("Loss"):
          cross_entropy = tf.losses.softmax_cross_entropy(logits=logits,
                                                          onehot_labels=labels)

          # Create a tensor named cross_entropy for logging purposes.
          tf.identity(cross_entropy, name='cross_entropy')

          # have to explicitly add weight decay, since
          # ImagenetModel does not set regularizers
          weight_decay = self.params.get('weight_decay', 1e-4)

          def exclude_batch_norm(name):
            return 'batch_normalization' not in name

          if self.params.get('regularize_bn', False):
            loss_filter_fn = lambda x: True
          else:
            loss_filter_fn = exclude_batch_norm

          # TODO: move this to regularizers somehow
          l2_loss = weight_decay * tf.add_n(
            # loss is computed using fp32 for numerical stability.
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in
             tf.trainable_variables() if loss_filter_fn(v.name)]
          )
          loss = cross_entropy + l2_loss
      else:
        deco_print("Inference Mode. Loss part of graph isn't built.")
        loss = None
      return loss, [logits]

  def maybe_print_logs(self, input_values, output_values):
    labels = input_values[1]
    logits = output_values[0]

    labels = np.where(labels == 1)[1]

    total = logits.shape[0]
    top1 = np.sum(np.argmax(logits, axis=1) == labels)
    top5 = np.sum(labels[:, np.newaxis] == np.argpartition(logits, -5)[:, -5:])

    top1 = 1.0 * top1 / total
    top5 = 1.0 * top5 / total
    deco_print("Train batch top-1: {:.4f}".format(top1), offset=4)
    deco_print("Train batch top-5: {:.4f}".format(top5), offset=4)
    return {
      "Train batch top-1": top1,
      "Train batch top-5": top5,
    }

  def finalize_evaluation(self, results_per_batch):
    top1 = 0.0
    top5 = 0.0
    total = 0.0

    for cur_total, cur_top1, cur_top5 in results_per_batch:
      top1 += cur_top1
      top5 += cur_top5
      total += cur_total

    top1 = 1.0 * top1 / total
    top5 = 1.0 * top5 / total
    deco_print("Validation top-1: {:.4f}".format(top1), offset=4)
    deco_print("Validation top-5: {:.4f}".format(top5), offset=4)
    return {
      "Eval top-1": top1,
      "Eval top-5": top5,
    }

  def evaluate(self, input_values, output_values):
    logits = output_values[0]
    labels = input_values[1]
    labels = np.where(labels == 1)[1]

    total = logits.shape[0]
    top1 = np.sum(np.argmax(logits, axis=1) == labels)
    top5 = np.sum(labels[:, np.newaxis] == np.argpartition(logits, -5)[:, -5:])
    return total, top1, top5
