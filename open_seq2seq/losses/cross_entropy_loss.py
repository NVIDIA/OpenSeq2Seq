# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from .loss import Loss


class CrossEntropyLoss(Loss):
  """Implementation of the usual cross_entropy loss with softmax."""

  def __init__(self, params, model, name="cross_entropy_loss"):
    super(CrossEntropyLoss, self).__init__(params, model, name)

  def _compute_loss(self, input_dict):
    logits = input_dict['decoder_output']['logits']
    labels = input_dict['target_tensors'][0]
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    return loss
