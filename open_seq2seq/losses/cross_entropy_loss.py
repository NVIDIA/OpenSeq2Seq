# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf

from .loss import Loss


def softmax_cross_entropy(
    onehot_labels, logits, weights=1.0, label_smoothing=0, scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
  if onehot_labels is None:
    raise ValueError("onehot_labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with tf.name_scope(scope, "softmax_cross_entropy_loss",
                     (logits, onehot_labels, weights)) as scope:
    logits = tf.convert_to_tensor(logits)
    onehot_labels = tf.cast(onehot_labels, logits.dtype)
    logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())

    num_classes = tf.cast(
        tf.shape(onehot_labels)[1], logits.dtype)
    smooth_positives = 1.0 - label_smoothing
    smooth_negatives = label_smoothing / num_classes
    onehot_labels = onehot_labels * smooth_positives + smooth_negatives

    onehot_labels = tf.stop_gradient(
        onehot_labels, name="labels_stop_gradient")
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=onehot_labels, logits=logits, name="xentropy")

    return tf.losses.compute_weighted_loss(
             losses, weights, scope, loss_collection, reduction=reduction)


class CrossEntropyLoss(Loss):
  @staticmethod
  def get_optional_params():
    return {
      'label_smoothing': float,
      'increase_smoothing_from_zero': bool,
    }

  def __init__(self, params, model, name="cross_entropy_loss"):
    super(CrossEntropyLoss, self).__init__(params, model, name)

  def _compute_loss(self, input_dict):
    logits = input_dict['decoder_output']['logits']
    labels = input_dict['target_tensors'][0]
    smoothing = self.params.get('label_smoothing', 0.0)
    if self.params.get('increase_smoothing_from_zero', False):
      global_step = tf.cast(tf.train.get_global_step(), tf.float32)
      smoothing = smoothing * global_step / (self._model.last_step - 1.0)
      if self._model.mode == 'train':
        tf.summary.scalar('label_smoothing', smoothing)

    if self.params.get('label_smoothing', 0.0) == 0 or self._model.mode == 'eval':
      loss = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=labels,
      )
    else:
      loss = softmax_cross_entropy(
        logits=logits,
        onehot_labels=labels,
        label_smoothing=smoothing,
      )
    return loss
