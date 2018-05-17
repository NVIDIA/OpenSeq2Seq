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



      if self.mode == "train" or self.mode == "eval":
        with tf.variable_scope("Loss"):
          loss = tf.losses.softmax_cross_entropy(logits=logits,
                                                 onehot_labels=labels)
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
