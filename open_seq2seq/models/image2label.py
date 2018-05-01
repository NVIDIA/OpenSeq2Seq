# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range
import tensorflow as tf
import numpy as np

from .model import Model
from open_seq2seq.utils.utils import deco_print

import sys
import os
sys.path.insert(0, os.path.abspath("tensorflow-models"))
from official.resnet.imagenet_main import ImagenetModel


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

    with tf.variable_scope("ForwardPass"):
      imagenet_model = ImagenetModel(resnet_size=self.params.get('resnet_size', 50))
      logits = imagenet_model(inputs, training=(self.mode == "train"))

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
      return loss, logits

  def maybe_print_logs(self, input_values, output_values):
    if self.on_horovod:
      labels = input_values[1]
      logits = output_values
    else:
      labels = input_values[1][0]
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

  def maybe_evaluate(self, inputs_per_batch, outputs_per_batch):
    top1 = 0.0
    top5 = 0.0
    total = 0.0

    last_cut = self.data_layer.get_size_in_samples() % \
               self.data_layer.params['batch_size']
    batch_idx = 0
    for input_values, output_values in zip(inputs_per_batch, outputs_per_batch):
      for gpu_id in range(self.num_gpus):
        logits = output_values[gpu_id]
        labels = input_values[1][gpu_id]
        labels = np.where(labels == 1)[1]
        # cutting last batch when dataset is not divisible by batch size
        # this assumes that num_gpus = 1 for now
        if batch_idx == len(inputs_per_batch) - 1:
          logits = logits[:last_cut]
          labels = labels[:last_cut]

        total += logits.shape[0]
        top1 += np.sum(np.argmax(logits, axis=1) == labels)
        top5 += np.sum(labels[:, np.newaxis] == np.argpartition(logits, -5)[:, -5:])
      batch_idx += 1

    top1 = 1.0 * top1 / total
    top5 = 1.0 * top5 / total
    deco_print("Validation top-1: {:.4f}".format(top1), offset=4)
    deco_print("Validation top-5: {:.4f}".format(top5), offset=4)
    return {
      "Eval top-1": top1,
      "Eval top-5": top5,
    }
