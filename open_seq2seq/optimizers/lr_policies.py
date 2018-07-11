# Copyright (c) 2017 NVIDIA Corporation
"""
Module containing various learning rate policies. Learning rate policy can
be any function that takes arbitrary arguments from the config (with additional
``global_step`` variable provided automatically) and returns learning rate
value for the current step.
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf


def fixed_lr(global_step, learning_rate):
  """Fixed learning rate policy.
  This function always returns ``learning_rate``, ignoring ``global_step``
  value.

  Args:
    global_step: global step TensorFlow tensor (ignored for this policy).
    learning_rate (float): fixed learning rate to use.

  Returns:
    learning rate at step ``global_step``.
  """
  return learning_rate


def piecewise_constant(global_step, learning_rate, boundaries,
                       decay_rates, steps_per_epoch=None):
  """Piecewise constant learning rate decay.
  When defined in the config, only ``boundaries`` and ``decay_rates`` need to
  be provided (other parameters are automatically populated by
  :class:`Model<models.model.Model>` class). ``boundaries`` are treated as
  epochs if ``num_epochs`` is provided in the config, otherwise treated as
  steps.

  Args:
    global_step: global step TensorFlow tensor.
    learning_rate (float): initial learning rate to use.
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
  vals = [learning_rate * decay for decay in decay_rates]
  return tf.train.piecewise_constant(global_step, boundaries, vals)


def exp_decay(global_step, learning_rate, decay_steps, decay_rate,
              use_staircase_decay, begin_decay_at=0, min_lr=0.0):
  """Exponential decay learning rate policy.
  This function is equivalent to ``tensorflow.train.exponential_decay`` with
  some additional functionality. Namely, it adds ``begin_decay_at`` parameter
  and ``min_lr`` parameter which are the first step to start decaying learning
  rate and minimal value of the learning rate correspondingly.

  Args:
    global_step: global step TensorFlow tensor.
    learning_rate (float): initial learning rate to use.
    decay_steps (int): number of steps to apply decay for.
    decay_rate (float): the rate of the decay.
    use_staircase_decay (bool): whether to use staircase decay.
    begin_decay_at (int): the first step to start decaying learning rate.
    min_lr (float): minimal value of the learning rate.

  Returns:
    learning rate at step ``global_step``.
  """
  new_lr = tf.cond(
      global_step < begin_decay_at,
      lambda: learning_rate,
      lambda: tf.train.exponential_decay(
          learning_rate,
          global_step - begin_decay_at,
          decay_steps,
          decay_rate,
          staircase=use_staircase_decay),
      name="learning_rate",
  )
  final_lr = tf.maximum(min_lr, new_lr)
  return final_lr


def poly_decay(global_step, learning_rate, decay_steps, power=1.0,
               begin_decay_at=0, min_lr=0.0):
  """Polynomial decay learning rate policy.
  This function is equivalent to ``tensorflow.train.polynomial_decay`` with
  some additional functionality. Namely, it adds ``begin_decay_at`` parameter
  which is the first step to start decaying learning rate.

  Args:
    global_step: global step TensorFlow tensor.
    learning_rate (float): initial learning rate to use.
    decay_steps (int): number of steps to apply decay for.
    power (float): power for polynomial decay.
    begin_decay_at (int): the first step to start decaying learning rate.
    min_lr (float): minimal value of the learning rate
        (same as ``end_learning_rate`` TensorFlow parameter).

  Returns:
    learning rate at step ``global_step``.
  """
  lr = tf.cond(
      global_step < begin_decay_at,
      lambda: learning_rate,
      lambda: tf.train.polynomial_decay(
          learning_rate,
          global_step=global_step-begin_decay_at,
          decay_steps=decay_steps,
          end_learning_rate=min_lr,
          power=power),
      name="learning_rate"
  )
  return lr


def transformer_policy(global_step, learning_rate, d_model, warmup_steps,
                       max_lr=None, coefficient=1.0, dtype=tf.float32):
  """Transformer's learning rate policy from
  https://arxiv.org/pdf/1706.03762.pdf
  with a hat (max_lr) (also called "noam" learning rate decay scheme).

  Args:
    global_step: global step TensorFlow tensor (ignored for this policy).
    learning_rate (float): initial learning rate to use.
    d_model (int): model dimensionality.
    warmup_steps (int): number of warm-up steps.
    max_lr (float): maximal learning rate, i.e. hat.
    coefficient (float): optimizer adjustment.
        Recommended 0.002 if using "Adam" else 1.0.
    dtype: dtype for this policy.

  Returns:
    learning rate at step ``global_step``.
  """
  step_num = tf.cast(global_step, dtype=dtype)
  ws = tf.cast(warmup_steps, dtype=dtype)

  decay = coefficient * d_model ** -0.5 * tf.minimum(
      (step_num + 1) * ws ** -1.5, (step_num + 1) ** -0.5
  )

  new_lr = decay * learning_rate
  if max_lr is not None:
    return tf.minimum(max_lr, new_lr)
  return new_lr
