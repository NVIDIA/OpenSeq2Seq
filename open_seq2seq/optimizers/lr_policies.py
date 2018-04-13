# Copyright (c) 2017 NVIDIA Corporation
"""
File containing various learning rate policies
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf


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
                       dtype=tf.float32):
  """
  Transformer's learning rate policy from https://arxiv.org/pdf/1706.03762.pdf
  :param learning_rate:
  :param var_global_step:
  :param d_model:
  :param warmup_steps:
  :param dtype:
  :return:
  """
  gs = tf.cast(var_global_step, dtype=dtype)
  ws = tf.cast(warmup_steps, dtype=dtype)
  t_lr = tf.scalar_mul(scalar=pow(1.0*d_model, -0.5),
                       x=tf.minimum(tf.pow(gs, -0.5),
                       gs*tf.pow(ws, -1.5)))
  if max_lr is not None:
    return tf.minimum(max_lr, t_lr)
  else:
    return t_lr

