# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Optimizer ops for use in layers and tf.learn."""

# This file was copy-pasted from TF repo on 10/04/2017 by Oleksii Kuchaiev
# The following changes were made:
# LARC support to "optimize_loss" function


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from six.moves import range

import six
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


from .automatic_loss_scaler import AutomaticLossScaler
from .mp_wrapper import MixedPrecisionOptimizerWrapper
from open_seq2seq.utils.utils import mask_nans, check_params


OPTIMIZER_CLS_NAMES = {
  "Adagrad": tf.train.AdagradOptimizer,
  "Adam": tf.train.AdamOptimizer,
  "Ftrl": tf.train.FtrlOptimizer,
  "Momentum": tf.train.MomentumOptimizer,
  "RMSProp": tf.train.RMSPropOptimizer,
  "SGD": tf.train.GradientDescentOptimizer,
}

OPTIMIZER_SUMMARIES = [
  "learning_rate",
  "gradients",
  "gradient_norm",
  "global_gradient_norm",
  "variables",
  "variable_norm",
  "larc_summaries",
]


# necessary to redefine this function for pure float16 support
def get_regularization_loss(scope=None, name="total_regularization_loss"):
  """Gets the total regularization loss.
  Args:
    scope: An optional scope name for filtering the losses to return.
    name: The name of the returned tensor.
  Returns:
    A scalar regularization loss.
  """
  losses = tf.losses.get_regularization_losses(scope)
  if losses:
    return tf.add_n(list(map(lambda x: tf.cast(x, tf.float32), losses)),
                    name=name)
  else:
    return tf.constant(0.0)


def reduce_gradients(grads_and_vars, on_horovod):
  if on_horovod:
    from horovod.common import size
    from horovod.tensorflow import allreduce

    if size() > 1:
      averaged_grads_and_vars = []
      with tf.name_scope("all_reduce"):
        for grad, var in grads_and_vars:
          if grad is not None:
            avg_grad = allreduce(grad)
            averaged_grads_and_vars.append((avg_grad, var))
          else:
            averaged_grads_and_vars.append((None, var))
      return averaged_grads_and_vars
    else:
      return grads_and_vars
  else:
    raise NotImplementedError("Reduce in tower-mode is not implemented.")


def optimize_loss(loss,
                  optimizer,
                  optimizer_params,
                  learning_rate_decay_fn,
                  dtype=tf.float32,
                  clip_gradients=None,
                  summaries=None,
                  larc_params=None,
                  loss_scaling=1.0,
                  on_horovod=False,
                  iter_size=1,
                  skip_update_ph=None):
  """Given loss and parameters for optimizer, returns a training op.

  Args:
    loss: Scalar `Tensor`.
    optimizer: string or class of optimizer, used as trainer.
        string should be name of optimizer, like 'SGD',
        'Adam', 'Adagrad'. Full list in OPTIMIZER_CLS_NAMES constant.
        class should be sub-class of `tf.Optimizer` that implements
        `compute_gradients` and `apply_gradients` functions.
    optimizer_params: parameters of the optimizer.
    dtype: model dtype (tf.float16, tf.float32 or "mixed").
    learning_rate_decay_fn: function, takes `global_step`
        `Tensor`s, returns `Tensor`.
        Can be used to implement any learning rate decay
        functions.
        For example: `tf.train.exponential_decay`.
        Ignored if `learning_rate` is not supplied.
    clip_gradients: float, max gradient norm to clip to.
    summaries: List of internal quantities to visualize on tensorboard. If not
        set only the loss and the learning rate will be reported. The
        complete list is in OPTIMIZER_SUMMARIES.
    larc_params: If not None, LARC re-scaling will
        be applied with corresponding parameters.
    loss_scaling: could be float or string. If float, static loss scaling
        is applied. If string, the corresponding automatic
        loss scaling algorithm is used. Must be one of 'Backoff'
        of 'LogMax' (case insensitive). Only used when dtype="mixed".
    on_horovod: whether the model is run on horovod.

  Returns:
    training op.
  """
  if summaries is None:
    summaries = ["learning_rate", "global_gradient_norm"]
  else:
    for summ in summaries:
      if summ not in OPTIMIZER_SUMMARIES:
        raise ValueError(
          "Summaries should be one of [{}], you provided {}.".format(
            ", ".join(OPTIMIZER_SUMMARIES), summ,
          ))
  if clip_gradients is not None and larc_params is not None:
    raise AttributeError(
      "LARC and gradient norm clipping should not be used together"
    )

  global_step = tf.train.get_or_create_global_step()
  lr = learning_rate_decay_fn(global_step)
  if "learning_rate" in summaries:
    tf.summary.scalar("learning_rate", lr)

  with tf.variable_scope("Loss_Optimization"):
    update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    loss = control_flow_ops.with_dependencies(list(update_ops), loss)

    # Create optimizer, given specified parameters.
    if isinstance(optimizer, six.string_types):
      if optimizer not in OPTIMIZER_CLS_NAMES:
        raise ValueError(
            "Optimizer name should be one of [{}], you provided {}.".format(
              ", ".join(OPTIMIZER_CLS_NAMES), optimizer
            ))
      optimizer = OPTIMIZER_CLS_NAMES[optimizer]
    opt = optimizer(learning_rate=lr, **optimizer_params)

    if isinstance(loss_scaling, six.string_types):
      loss_scaling = AutomaticLossScaler(algorithm=loss_scaling)

    if dtype == 'mixed':
      opt = MixedPrecisionOptimizerWrapper(opt, loss_scale=loss_scaling)

    # Compute gradients.
    grads_and_vars = opt.compute_gradients(
      loss, colocate_gradients_with_ops=True,
    )

    if on_horovod:
      if iter_size > 1:
        grads_and_vars_accum = []
        accum_ops = []
        for grad, var in grads_and_vars:
          # necessary to use tf.Variable directly to instantiate cudnn rnn cells
          # which don't have explicit shape.
          grad_accum = tf.Variable(
            initial_value=tf.zeros_like(var),
            name=grad.name.split(":")[0] + "_accum",
            expected_shape=var.shape,
            dtype=grad.dtype,
            trainable=False,
            validate_shape=bool(var.get_shape())
          )
          if isinstance(grad, tf.IndexedSlices):
            add_grads = tf.scatter_nd_add(grad_accum, grad.indices,
                                          grad.values / iter_size)
          else:
            add_grads = grad_accum + grad / iter_size

          accum_ops.append(tf.assign(grad_accum, add_grads))
          grads_and_vars_accum.append((grad_accum, var))

        accum_op = tf.group(accum_ops)

        def update_and_clear_op():
          with tf.control_dependencies([accum_op]):
            red_grad_updates = opt.apply_gradients(
              post_process_gradients(
                reduce_gradients(grads_and_vars_accum, on_horovod=True),
                lr=lr,
                clip_gradients=clip_gradients,
                larc_params=larc_params,
                summaries=summaries,
              ),
              global_step=global_step,
            )

          with tf.control_dependencies([red_grad_updates]):
            return tf.group([tf.assign(g, tf.zeros_like(g))
                             for g, v in grads_and_vars_accum])

        grad_updates = tf.cond(
          pred=skip_update_ph,
          true_fn=lambda: accum_op,
          false_fn=update_and_clear_op,
        )
      else:
        grad_updates = opt.apply_gradients(
          post_process_gradients(
            reduce_gradients(grads_and_vars, on_horovod=True),
            lr=lr,
            clip_gradients=clip_gradients,
            larc_params=larc_params,
            summaries=summaries,
          ),
          global_step=global_step,
        )
    else:
      grad_updates = opt.apply_gradients(
        post_process_gradients(
          grads_and_vars,
          lr=lr,
          clip_gradients=clip_gradients,
          larc_params=larc_params,
          summaries=summaries,
        ),
        global_step=global_step,
      )

    # Ensure the train_tensor computes grad_updates.
    train_tensor = control_flow_ops.with_dependencies([grad_updates], loss)

    return train_tensor


def post_process_gradients(grads_and_vars, summaries, lr,
                           clip_gradients, larc_params):
  """Applies post processing to gradients, i.e. clipping, LARC, summaries."""
  if "global_gradient_norm" in summaries:
    tf.summary.scalar(
      "global_gradient_norm",
      _global_norm_with_cast(grads_and_vars),
    )

  # Optionally clip gradients by global norm.
  if clip_gradients is not None:
    grads_and_vars = _clip_gradients_by_norm(grads_and_vars, clip_gradients)

  # Add histograms for variables, gradients and gradient norms.
  for gradient, variable in grads_and_vars:
    if isinstance(gradient, tf.IndexedSlices):
      grad_values = gradient.values
    else:
      grad_values = gradient

    if isinstance(variable, tf.IndexedSlices):
      var_values = variable.values
    else:
      var_values = variable

    if grad_values is not None:
      var_name = variable.name.replace(":", "_")
      if "gradients" in summaries:
        # need to mask nans for automatic loss scaling
        tf.summary.histogram("gradients/%s" % var_name, mask_nans(grad_values))
      if "gradient_norm" in summaries:
        tf.summary.scalar("gradient_norm/%s" % var_name, tf.norm(grad_values))
      if "variables" in summaries:
        tf.summary.histogram("variables/%s" % var_name, var_values)
      if "variable_norm" in summaries:
        tf.summary.scalar("variable_norm/%s" % var_name, tf.norm(var_values))

  if clip_gradients is not None and "global_gradient_norm" in summaries:
    tf.summary.scalar(
      "global_clipped_gradient_norm",
      _global_norm_with_cast(grads_and_vars),
    )

  # LARC gradient re-scaling
  if larc_params is not None:
    check_params(
      config=larc_params,
      required_dict={'larc_eta': float},
      optional_dict={
        'larc_mode': ['clip', 'scale'],
        'min_update': float,
        'epsilon': float
      },
    )
    larc_eta = larc_params['larc_eta']
    larc_mode = larc_params.get('larc_mode', 'clip')
    min_update = larc_params.get('min_update', 1e-7)
    eps = larc_params.get('epsilon', 1e-7)

    grads_and_vars_larc = [None] * len(grads_and_vars)
    for idx, (g, v) in enumerate(grads_and_vars):
      var_dtype = v.dtype
      v_norm = tf.norm(tensor=tf.cast(v, tf.float32), ord=2)
      g_norm = tf.norm(tensor=tf.cast(g, tf.float32), ord=2)

      if larc_mode == 'clip':
        larc_grad_update = tf.maximum(
          larc_eta * v_norm / (lr * (g_norm + eps)),
          min_update,
        )
        if "larc_summaries" in summaries:
          tf.summary.scalar('larc_clip_on/{}'.format(v.name),
                            tf.cast(tf.less(larc_grad_update, 1.0), tf.int32))
        larc_grad_update = tf.minimum(larc_grad_update, 1.0)
      else:
        larc_grad_update = tf.maximum(
          larc_eta * v_norm / (g_norm + eps),
          min_update,
        )
      larc_grad_update = tf.saturate_cast(larc_grad_update, var_dtype)
      grads_and_vars_larc[idx] = (larc_grad_update * g, v)

      # adding additional summary
      if "larc_summaries" in summaries:
        tf.summary.scalar('larc_grad_update/{}'.format(v.name),
                          larc_grad_update)
        tf.summary.scalar("larc_final_lr/{}".format(v.name),
                          tf.cast(lr, var_dtype) * larc_grad_update)
    grads_and_vars = grads_and_vars_larc
  return grads_and_vars


def _global_norm_with_cast(grads_and_vars):
  return tf.global_norm(list(map(
    lambda x: tf.cast(x, tf.float32),
    list(zip(*grads_and_vars))[0])
  ))


def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
  """Clips gradients by global norm."""
  gradients, variables = zip(*grads_and_vars)
  clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients)
  return list(zip(clipped_gradients, variables))
