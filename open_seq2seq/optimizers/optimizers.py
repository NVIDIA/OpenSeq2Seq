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

import collections
import six
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from open_seq2seq.utils.utils import mask_nans, check_params
from .automatic_loss_scaler import AutomaticLossScaler
from .mp_wrapper import MixedPrecisionOptimizerWrapper

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
    "loss_scale"
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


def reduce_gradients(grads_and_vars, on_horovod, model=None):
  if on_horovod:
    from horovod.tensorflow import allreduce, size

    if size() > 1:
      averaged_grads_and_vars = []
      with tf.name_scope("all_reduce"):
        for grad, var in grads_and_vars:
          if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
              if model._decoder.params.get('shared_embed', False):
                from tensorflow.python.training.optimizer import _deduplicate_indexed_slices
                summed_values, unique_indices = _deduplicate_indexed_slices(
                    values=grad.values, indices=grad.indices)
                gradient_no_duplicate_indices = tf.IndexedSlices(
                    indices=unique_indices,
                    values=summed_values,
                    dense_shape=grad.dense_shape)
                grad = tf.convert_to_tensor(gradient_no_duplicate_indices)
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
                  var_list=None,
                  dtype=tf.float32,
                  clip_gradients=None,
                  summaries=None,
                  larc_params=None,
                  loss_scaling=1.0,
                  loss_scaling_params=None,
                  on_horovod=False,
                  iter_size=1,
                  skip_update_ph=None,
                  model=None):
  """Given loss and parameters for optimizer, returns a training op.

  Args:
    loss: Scalar `Tensor`.
    optimizer: string or class of optimizer, used as trainer.
        string should be name of optimizer, like 'SGD',
        'Adam', 'Adagrad'. Full list in OPTIMIZER_CLS_NAMES constant.
        class should be sub-class of `tf.Optimizer` that implements
        `compute_gradients` and `apply_gradients` functions.
    optimizer_params: parameters of the optimizer.
    var_list: List of trainable variables. Can be used to freeze
        certain trainable variables by excluding them from this list. 
        If set to None, all trainable variables will be optimized.
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
    summaries = ["learning_rate", "global_gradient_norm", "loss_scale"]
  else:
    for summ in summaries:
      if summ not in OPTIMIZER_SUMMARIES:
        raise ValueError(
            "Summaries should be one of [{}], you provided {}.".format(
                ", ".join(OPTIMIZER_SUMMARIES), summ,
            )
        )
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
            )
        )
      optimizer = OPTIMIZER_CLS_NAMES[optimizer]
    opt = optimizer(learning_rate=lr, **optimizer_params)

    if isinstance(loss_scaling, six.string_types):
      loss_scaling = AutomaticLossScaler(
          algorithm=loss_scaling,
          params=loss_scaling_params
      )
      if "loss_scale" in summaries:
        tf.summary.scalar("loss_scale", loss_scaling.loss_scale)

    if dtype == 'mixed':
      opt = MixedPrecisionOptimizerWrapper(opt, loss_scale=loss_scaling)

    # Compute gradients.
    grads_and_vars = opt.compute_gradients(
        loss, colocate_gradients_with_ops=True, var_list=var_list
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
                    reduce_gradients(grads_and_vars_accum, on_horovod=True, model=model),
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
                reduce_gradients(grads_and_vars, on_horovod=True, model=model),
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
      list(zip(*grads_and_vars))[0]
  )))


def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
  """Clips gradients by global norm."""
  gradients, variables = zip(*grads_and_vars)
  dtypes = [var.dtype for var in variables]

  # Clip gradients in float32
  clipped_gradients, _ = _clip_by_global_norm(
      gradients,
      clip_gradients,
      use_norm=_global_norm_with_cast(grads_and_vars)
  )

  # Convert gradients back to the proper dtype
  clipped_gradients = [
      tf.cast(grad, dtype)
      for grad, dtype in zip(clipped_gradients, dtypes)
  ]

  return list(zip(clipped_gradients, variables))

def _clip_by_global_norm(t_list, clip_norm, use_norm, name=None):
  """Clips values of multiple tensors by the ratio of the sum of their norms.
  Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`,
  this operation returns a list of clipped tensors `list_clipped`
  and the global norm (`global_norm`) of all tensors in `t_list`. The global
  norm is expected to be pre-computed and passed as use_norm.
  To perform the clipping, the values `t_list[i]` are set to:
      t_list[i] * clip_norm / max(global_norm, clip_norm)
  where:
      global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
  If `clip_norm > global_norm` then the entries in `t_list` remain as they are,
  otherwise they're all shrunk by the global ratio.
  Any of the entries of `t_list` that are of type `None` are ignored.
  This is the correct way to perform gradient clipping (for example, see
  [Pascanu et al., 2012](http://arxiv.org/abs/1211.5063)
  ([pdf](http://arxiv.org/pdf/1211.5063.pdf))).
  However, it is slower than `clip_by_norm()` because all the parameters must be
  ready before the clipping operation can be performed.

  Args:
    t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
    clip_norm: A 0-D (scalar) `Tensor` > 0. The clipping ratio.
    use_norm: A 0-D (scalar) `Tensor` of type `float` (optional). The global
      norm to use. If not provided, `global_norm()` is used to compute the norm.
    name: A name for the operation (optional).

  Returns:
    list_clipped: A list of `Tensors` of the same type as `list_t`.
    global_norm: A 0-D (scalar) `Tensor` representing the global norm.

  Raises:
    TypeError: If `t_list` is not a sequence.
  """
  if (not isinstance(t_list, collections.Sequence)
      or isinstance(t_list, six.string_types)):
    raise TypeError("t_list should be a sequence")
  t_list = list(t_list)

  # Removed as use_norm should always be passed
  # if use_norm is None:
  #   use_norm = global_norm(t_list, name)

  with tf.name_scope(name, "clip_by_global_norm",
                     t_list + [clip_norm]) as name:
    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    scale = clip_norm * tf.minimum(
        1.0 / use_norm,
        tf.ones([1], dtype=use_norm.dtype) / clip_norm)

    values = [
        tf.cast(
            tf.convert_to_tensor(
                t.values if isinstance(t, tf.IndexedSlices) else t,
                name="t_%d" % i),
            dtype=tf.float32
        )
        if t is not None else t
        for i, t in enumerate(t_list)]

    values_clipped = []
    for i, v in enumerate(values):
      if v is None:
        values_clipped.append(None)
      else:
        with tf.colocate_with(v):
          values_clipped.append(
              tf.identity(v * scale, name="%s_%d" % (name, i)))

    list_clipped = [
        tf.IndexedSlices(c_v, t.indices, t.dense_shape)
        if isinstance(t, tf.IndexedSlices)
        else c_v
        for (c_v, t) in zip(values_clipped, t_list)]

  return list_clipped, use_norm
  
