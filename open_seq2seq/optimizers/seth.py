# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of Stochastic Average Gradient."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.train import MomentumOptimizer
import tensorflow as tf

class SethOptimizer(MomentumOptimizer):
  """Optimizer that implements the Layerwise ADAM

    ```
    m_t <- beta * m_{t-1} + (1 - beta) * g_t
    variable_t <- variable_{t-1} - lr_t * m_t
    ```

  """

  def __init__(self, learning_rate, momentum, epsilon = 1e-5,
               use_locking=False, name='SethOptimizer', use_nesterov=False):
    """Constructs a new SethOptimizer

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      momentum: A `Tensor` or a floating point value.  The momentum.
      epsilon:  A `Tensor` or a floating point value.  Default = 1e-5.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "SethOptimizer".
      use_nesterov: If `True` use Nesterov Momentum.

    """
    super(SethOptimizer, self).__init__(learning_rate, momentum,
                                       use_locking, name, use_nesterov)
    self._beta = momentum
    self._epsilon = epsilon
    # Tensor versions of the constructor arguments, created in _prepare().
    self._beta_t = None

  # def apply_gradients(self, grads_and_vars, global_step=None, name=None):
  #   self._beta_t = ops.convert_to_tensor(self._beta, name='beta')
  #   grad_vars_s = [(g * math_ops.cast(self._beta_t, g.dtype.base_dtype), v) \
  #                          for (g, v) in grads_and_vars]
  #   return super(SagOptimizer, self).apply_gradients(
  #     grad_vars_s, global_step=global_step, name=name)


  def _prepare(self):
    self._beta_t = ops.convert_to_tensor(self._beta, name='beta')
    super(SethOptimizer, self)._prepare()

  def _apply_dense(self, grad, var):
    beta_t = tf.cast(self._beta_t, grad.dtype.base_dtype)
    g_norm = tf.norm(tensor=tf.cast(grad, tf.float32), ord=2)
    grad_s = tf.scalar_mul((beta_t/(g_norm + self._epsilon)), grad)
    return super(SethOptimizer, self)._apply_dense(grad_s, var)

  def _apply_sparse(self, grad, var):
    beta_t = tf.cast(self._beta_t, grad.dtype.base_dtype)
    g_norm = tf.norm(tensor=tf.cast(grad, tf.float32), ord=2)
    grad_s = tf.scalar_mul((beta_t/(g_norm + self._epsilon)), grad)
    return super(SethOptimizer, self)._apply_sparse(grad_s, var)