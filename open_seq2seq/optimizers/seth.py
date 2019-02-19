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
    grad = tf.scalar_mul((beta_t/(g_norm + self._epsilon)), grad)
    return super(SethOptimizer, self)._apply_dense(grad, var)

  def _apply_sparse(self, grad, var):
    beta_t = tf.cast(self._beta_t, grad.dtype.base_dtype)
    g_norm = tf.norm(tensor=tf.cast(grad, tf.float32), ord=2)
    grad = tf.scalar_mul((beta_t/(g_norm + self._epsilon)), grad)
    return super(SethOptimizer, self)._apply_sparse(grad, var)

#---------------------------------------------------------------------

class NoahOptimizer(MomentumOptimizer):
  """Optimizer that implements algorithm "similar" to Layerwise ADAM

    ```
     Layer-wise ema of grad norms  - has length = num_of_layers
       u_t <-- beta1 * u_{t-1} + (1 - beta1) * |g_t|

    1st order momentum - has lenghts = num of vars:
       m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t / u_t

    Weight update:
       w_t <- w_{t-1} - lr_t * m_t /
    ```

  """

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.95,
               epsilon=1e-6, ord =2,
               use_locking=False, name='NoahOptimizer'):
    """Constructs a new NoahOptimizer

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      beta1: A `Tensor` or a floating point value, used in ema for 1st order momentum.
      beta2: A `Tensor` or a floating point value, used in ema for layer grad norms,
      epsilon: a floating point value.  Default = 1e-6.
      ord: a floating point value. Ord if grad norms  Default = 2.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "SethOptimizer".
      use_nesterov: If `True` use Nesterov Momentum.

    """
    super(NoahOptimizer, self).__init__(learning_rate, momentum=beta1,
                                        use_locking=use_locking, name=name,
                                        use_nesterov=False)
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    self._ord = ord
    # Tensor versions, converted to tensors in apply_gradients
    self._beta1_t = None
    self._beta2_t = None
    self._grads_ema = None

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    self._beta1_t = ops.convert_to_tensor(self._beta1, name='beta1')
    self._beta2_t = ops.convert_to_tensor(self._beta2, name='beta2')
    len_vars=len(grads_and_vars)

    if self._grads_ema is None:
      self._grads_ema = [None] * len_vars
      for i in range(len_vars):
        self._grads_ema[i] = tf.get_variable(name="grad_ema"+str(i),
                                    shape=[], dtype=tf.float32,
                                    initializer=tf.keras.initializers.Zeros(),
                                    trainable=False)

    for i, (grad, var) in enumerate(grads_and_vars):
      g_norm=tf.norm(tensor=tf.cast(grad, tf.float32), ord=self._ord)
      self._grads_ema[i] = tf.cond(tf.equal(self._grads_ema[i], 0.),
                  lambda: g_norm,
                  lambda: self._grads_ema[i]*self._beta2_t + g_norm*(1.-self._beta2_t)
                                 )
      g_factor = self._beta1_t / (self._grads_ema[i] + self._epsilon)
      grad = tf.scalar_mul(g_factor, grad)
      grads_and_vars[i] = (grad, var)

    return super(NoahOptimizer, self).apply_gradients(
      grads_and_vars, global_step=global_step, name=name)


  # def _prepare(self):
  #   self._beta1_t = ops.convert_to_tensor(self._beta1, name='beta1')
  #   self._beta2_t = ops.convert_to_tensor(self._beta2, name='beta2')
  #   super(NoahOptimizer, self)._prepare()
  #
  # def _apply_dense(self, grad, var):
  #   beta1_t = tf.cast(self._beta1_t, grad.dtype.base_dtype)
  #   beta2_t = tf.cast(self._beta2_t, grad.dtype.base_dtype)
  #   g_norm = tf.norm(tensor=tf.cast(grad, tf.float32), ord=2)
  #
  #   grad_s = tf.scalar_mul((beta_t / (g_norm + self._epsilon)), grad)
  #   return super(NoahOptimizer, self)._apply_dense(grad_s, var)
  #
  # def _apply_sparse(self, grad, var):
  #   beta_t = tf.cast(self._beta_t, grad.dtype.base_dtype)
  #   g_norm = tf.norm(tensor=tf.cast(grad, tf.float32), ord=2)
  #   grad_s = tf.scalar_mul((beta_t / (g_norm + self._epsilon)), grad)
  #   return super(NoahOptimizer, self)._apply_sparse(grad_s, var)