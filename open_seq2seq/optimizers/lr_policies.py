# Copyright (c) 2017 NVIDIA Corporation
"""
File containing various learning rate policies
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf


def piecewise_constant(initial_learning_rate, global_step, boundaries,
                       decay_rates, steps_per_epoch=None):
  """Piecewise constant learning rate decay.
  When defined in the config, only ``boundaries`` and ``decay_rates`` need to
  be provided (other parameters are automatically populated by
  :class:`Model<models.model.Model>` class). ``boundaries`` are treated as
  epochs if ``num_epochs`` is provided in the config, otherwise treated as
  steps.

  Args:
    initial_learning_rate (float): initial learning rate to use.
    global_step: global step TensorFlow tensor.
    boundaries (list): could be either defined in steps
        (if ``batches_per_epoch=None``) or in epochs if ``batches_per_epoch``
        parameter is defined.
    decay_rates: multiplier of the initial learning rate for each boundary.
    steps_per_epoch: number of batches in one training epoch. If provided,
        boundaries are treated as epochs, otherwise as steps.

  Returns:
    learning rate at step ``global_step``.
  """
  if steps_per_epoch is not None:
    boundaries = [steps_per_epoch * epoch for epoch in boundaries]
  decay_rates = [1.0] + decay_rates
  vals = [initial_learning_rate * decay for decay in decay_rates]
  return tf.train.piecewise_constant(global_step, boundaries, vals)


def exp_decay(learning_rate,
              var_global_step,
              begin_decay_at,
              decay_steps,
              decay_rate,
              use_staircase_decay,
              min_lr):
  """
  Exponential decay
  :param learning_rate:
  :param var_global_step:
  :param begin_decay_at:
  :param decay_steps:
  :param decay_rate:
  :param use_staircase_decay:
  :param min_lr:
  :return:
  """
  new_lr = tf.cond(
    var_global_step < begin_decay_at,
    lambda: learning_rate,
    lambda: tf.train.exponential_decay(
      learning_rate,
      var_global_step - begin_decay_at,
      decay_steps,
      decay_rate,
      staircase=use_staircase_decay),
    name="learning_rate",
  )
  final_lr = tf.maximum(min_lr, new_lr)
  return final_lr


def poly_decay(learning_rate,
               var_global_step,
               decay_steps,
               begin_decay_at=0,
               power=1.0,
               min_lr=0.0):
  """
  Polynomial decay
  :param learning_rate:
  :param var_global_step:
  :param decay_steps:
  :param begin_decay_at:
  :param power:
  :param min_lr:
  :return:
  """
  lr = tf.cond(
    var_global_step < begin_decay_at,
    lambda: learning_rate,
    lambda: tf.train.polynomial_decay(
      learning_rate,
      global_step=var_global_step-begin_decay_at,
      decay_steps=decay_steps,
      end_learning_rate=min_lr,
      power=power),
    name="learning_rate"
  )
  return lr


def transformer_policy(learning_rate,
                       var_global_step,
                       d_model,
                       warmup_steps,
                       max_lr=None,
                       coefficient=1.0,
                       dtype=tf.float32):
  """
  Transformer's learning rate policy from https://arxiv.org/pdf/1706.03762.pdf,
  with a hat (max_lr).
  (Also called "noam" learning_rate_decay_scheme)
  :param learning_rate: learning_rate
  :param var_global_step: global step
  :param d_model: model dimensionality
  :param warmup_steps: steps to do warmup
  :param max_lr: max_lr, e.g. hat
  :param coefficient: optimizer adjustment. Recommended 0.002 if "Adam" else 1.0
  :param dtype:
  :return: adjusted learning rate
  """
  step_num = tf.cast(var_global_step, dtype=dtype)
  ws = tf.cast(warmup_steps, dtype=dtype)

  decay = coefficient * d_model ** -0.5 * tf.minimum(
    (step_num + 1) * ws ** -1.5, (step_num + 1) ** -0.5)

  new_lr = decay * learning_rate
  if max_lr is not None:
    return tf.minimum(max_lr, new_lr)
  else:
    return new_lr

