# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

from open_seq2seq.utils.utils import deco_print
from .encoder_decoder import EncoderDecoderModel


class Image2Label(EncoderDecoderModel):
  def maybe_print_logs(self, input_values, output_values, training_step):
    labels = input_values['target_tensors'][0]
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

  def finalize_evaluation(self, results_per_batch, training_step=None):
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
    labels = input_values['target_tensors'][0]
    labels = np.where(labels == 1)[1]

    total = logits.shape[0]
    top1 = np.sum(np.equal(np.argmax(logits, axis=1), labels))
    top5 = np.sum(np.equal(labels[:, np.newaxis],
                           np.argpartition(logits, -5)[:, -5:]))
    return total, top1, top5

  def _get_num_objects_per_step(self, worker_id=0):
    """Returns number of images in current batch, i.e. batch size."""
    data_layer = self.get_data_layer(worker_id)
    num_images = tf.shape(data_layer.input_tensors['source_tensors'][0])[0]
    return num_images
