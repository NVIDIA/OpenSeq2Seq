# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from open_seq2seq.utils.utils import check_params

class AutomaticLossScaler(object):
  SUPPORTED_ALGOS = ['backoff', 'logmax']

  def __init__(self, algorithm='Backoff', params=None):
    algorithm = algorithm.lower().strip()
    if algorithm == 'backoff':
      self.scaler = BackoffScaler(params)
    elif algorithm == 'logmax':
      self.scaler = LogMaxScaler(params)  # ppf(.999)
    else:
      raise ValueError('Unknown scaling algorithm: {}'.format(algorithm))

  def update_op(self, has_nan, amax):
    return self.scaler.update_op(has_nan, amax)

  @property
  def loss_scale(self):
    return self.scaler.loss_scale

  @staticmethod
  def check_grads(grads_and_vars):
    has_nan_ops = []
    amax_ops = []

    for grad, _ in grads_and_vars:
      if grad is not None:
        if isinstance(grad, tf.IndexedSlices):
          x = grad.values
        else:
          x = grad

        has_nan_ops.append(tf.reduce_any(tf.is_nan(x)))
        amax_ops.append(tf.reduce_max(tf.abs(x)))

    has_nan = tf.reduce_any(has_nan_ops)
    amax = tf.reduce_max(amax_ops)
    return has_nan, amax


class BackoffScaler(object):
  def __init__(self, params):
    if params is None:
      params = {}
    check_params(
        config=params,
        required_dict={},
        optional_dict={
            'scale_min': float,
            'scale_max': float,
            'step_factor': float,
            'step_window': int
        },
    )
    self.scale_min = params.get('scale_min', 1.0)
    self.scale_max = params.get('scale_max', 2.**24)
    self.step_factor = params.get('step_factor', 2.0)
    self.step_window = params.get('step_window', 2000)

    self.iteration = tf.Variable(initial_value=0,
                                 trainable=False,
                                 dtype=tf.int64)
    self.last_overflow_iteration = tf.Variable(initial_value=-1,
                                               trainable=False,
                                               dtype=tf.int64)
    self.scale = tf.Variable(initial_value=self.scale_max,
                             trainable=False)

  def update_op(self, has_nan, amax):
    def overflow_case():
      new_scale_val = tf.clip_by_value(self.scale / self.step_factor,
                                       self.scale_min, self.scale_max)
      scale_assign = tf.assign(self.scale, new_scale_val)
      overflow_iter_assign = tf.assign(self.last_overflow_iteration,
                                       self.iteration)
      with tf.control_dependencies([scale_assign, overflow_iter_assign]):
        return tf.identity(self.scale)

    def scale_case():
      since_overflow = self.iteration - self.last_overflow_iteration
      should_update = tf.equal(since_overflow % self.step_window, 0)
      def scale_update_fn():
        new_scale_val = tf.clip_by_value(self.scale * self.step_factor,
                                         self.scale_min, self.scale_max)
        return tf.assign(self.scale, new_scale_val)
      return tf.cond(should_update,
                     scale_update_fn,
                     lambda: self.scale)

    iter_update = tf.assign_add(self.iteration, 1)
    overflow = tf.logical_or(has_nan, tf.is_inf(amax))

    update_op = tf.cond(overflow,
                        overflow_case,
                        scale_case)
    with tf.control_dependencies([update_op]):
      return tf.identity(iter_update)

  @property
  def loss_scale(self):
    return self.scale


class LogMaxScaler(object):
  def __init__(self, params):
    if params is None:
      params = {}
    check_params(
        config=params,
        required_dict={},
        optional_dict={
            'scale_min': float,
            'scale_max': float,
            'log_max': float,
            'beta1': float,
            'beta2': float,
            'overflow_std_dev': float
        },
    )
    self.scale_min = params.get('scale_min', 1.0)
    self.scale_max = params.get('scale_max', 2.**24)
    self.log_max = params.get('log_max', 16.)
    self.beta1 = params.get('beta1', 0.99)
    self.beta2 = params.get('beta2', 0.999)
    self.overflow_std_dev = params.get('overflow_std_dev', 3.09)

    self.iteration = tf.Variable(initial_value=0,
                                 trainable=False,
                                 dtype=tf.int64)
    self.scale = tf.Variable(initial_value=1.0,
                             trainable=False)
    self.x_hat = tf.Variable(initial_value=0,
                             trainable=False,
                             dtype=tf.float32)
    self.slow_x_hat = tf.Variable(initial_value=0,
                                  trainable=False,
                                  dtype=tf.float32)
    self.xsquared_hat = tf.Variable(initial_value=0,
                                    trainable=False,
                                    dtype=tf.float32)
    self.b1_correction = tf.Variable(initial_value=1.,
                                     trainable=False,
                                     dtype=tf.float32)
    self.b2_correction = tf.Variable(initial_value=1.,
                                     trainable=False,
                                     dtype=tf.float32)

  # NB: assumes that `amax` is already has been downscaled
  def update_op(self, has_nan, amax):
    is_nonfinite = tf.logical_or(has_nan, tf.is_inf(amax))
    x = tf.cond(is_nonfinite,
                lambda: tf.pow(2., self.log_max),
                lambda: tf.log(amax) / tf.log(tf.constant(2.)))

    x_hat_assn = tf.assign(self.x_hat, self.beta1 * self.x_hat +
                           (1 - self.beta1) * x)
    b1_corr_assn = tf.assign(self.b1_correction,
                             self.b1_correction * self.beta1)
    with tf.control_dependencies([x_hat_assn, b1_corr_assn]):
      mu = self.x_hat.read_value() / (1 - self.b1_correction.read_value())

    slow_x_hat_assn = tf.assign(self.slow_x_hat, self.beta2 * self.slow_x_hat +
                                (1 - self.beta2) * x)
    xsquared_hat_assn = tf.assign(
        self.xsquared_hat,
        self.beta2 * self.xsquared_hat + (1 - self.beta2) * (x * x),
    )
    b2_corr_assn = tf.assign(self.b2_correction,
                             self.b2_correction * self.beta2)
    with tf.control_dependencies([slow_x_hat_assn, xsquared_hat_assn,
                                  b2_corr_assn]):
      e_xsquared = self.xsquared_hat.read_value() / \
                   (1 - self.b2_correction.read_value())
      slow_mu = self.slow_x_hat.read_value() / \
                (1 - self.b2_correction.read_value())

    sigma2 = e_xsquared - (slow_mu * slow_mu)
    sigma = tf.sqrt(tf.maximum(sigma2, tf.constant(0.)))

    log_cutoff = sigma * self.overflow_std_dev + mu
    log_difference = 16 - log_cutoff
    proposed_scale = tf.pow(2., log_difference)
    scale_update = tf.assign(
        self.scale,
        tf.clip_by_value(proposed_scale, self.scale_min, self.scale_max),
    )
    iter_update = tf.assign_add(self.iteration, 1)

    with tf.control_dependencies([scale_update]):
      return tf.identity(iter_update)

  @property
  def loss_scale(self):
    return self.scale
