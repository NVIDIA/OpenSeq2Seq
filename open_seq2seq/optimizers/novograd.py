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

class NovoGrad(MomentumOptimizer):
  """
  Optimizer that implements Stochastic Average Gradient(SAG)
  with first momentum = ema of layer-wise normalized gradients,
  when normalization is done by ema of norm for layer gradients

    ```
    Layer-wise momentum for norm of grads:
       u_t <-- beta1 * u_{t-1} + (1 - beta1) * |g_t|

    momentum = ema of grads normalized by u_t:
       m_t <- beta1 * m_{t-1} + (1 - beta1) * ( g_t / u_t )

    ```

  """

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.95,
               epsilon=1e-6, ord =2, use_locking=False, name='NovoGrad'):
    """Constructs a new NovoGrad

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      beta1: A `Tensor` or a float, used in ema for momentum.
      beta2: A `Tensor` or a float, used in ema for grad norms,
      epsilon: a float.  Default = 1e-6.
      use_locking: If `True` use locks for update operations.
      name: Optional, name prefix for the ops created when applying
        gradients.  Defaults to "NovoGrad".
      use_nesterov: If `True` use Nesterov Momentum.

    """
    super(NovoGrad, self).__init__(learning_rate, momentum=beta1,
                 use_locking=use_locking, name=name, use_nesterov=False)
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
    # init ema variables if required
    if self._grads_ema is None:
      self._grads_ema = [None] * len_vars
      for i in range(len_vars):
        self._grads_ema[i] = tf.get_variable(name="novograd_ema"+str(i),
                                    shape=[], dtype=tf.float32,
                                    initializer=tf.keras.initializers.Zeros(),
                                    trainable=False)
    # compute ema for each layer
    for i, (grad, var) in enumerate(grads_and_vars):
      g_norm=tf.norm(tensor=tf.cast(grad, tf.float32), ord=self._ord)
      self._grads_ema[i] = tf.cond(tf.equal(self._grads_ema[i], 0.),
             lambda: g_norm,
             lambda: self._grads_ema[i]*self._beta2_t+g_norm*(1.-self._beta2_t)
                                 )
      g_factor = self._beta1_t / (self._grads_ema[i] + self._epsilon)
      grad = tf.scalar_mul(g_factor, grad)
      grads_and_vars[i] = (grad, var)

    return super(NovoGrad, self).apply_gradients(
      grads_and_vars, global_step=global_step, name=name)

#-----------------------------------------------------------------------------
class NovoGrad2(MomentumOptimizer):
  """
  Optimizer that implements Stochastic Average Gradient(SAG)
  with first momentum = ema of layer-wise normalized gradients,
  when normalization is done by sqrt(ema of sqr(grads)),
  similar to ADAM

    ```
    ema of Layer-wise norm of grads:
       v_t <-- beta1 * v_{t-1} + (1 - beta1) * (g_t)^2

    momentum = ema of grads normalized by u_t:
       m_t <- beta1 * m_{t-1} + (1 - beta1) * (g_t / sqrt(v_t+epsilon)

    Weight update:
       w_t <- w_{t-1} - lr_t * m_t
    ```

  """

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.95,
               epsilon=1e-6, ord=2, use_locking=False, name='NovoGrad2', weight_decay=0.0):
    """Constructs a new NovoGrad

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      beta1: A `Tensor` or a float, used in ema for momentum.
      beta2: A `Tensor` or a float, used in ema for grad norms,
      epsilon: a float.  Default = 1e-6.
      use_locking: If `True` use locks for update operations.
      name: Optional, name prefix for the ops created when applying
        gradients.  Defaults to "NovoGrad".
      use_nesterov: If `True` use Nesterov Momentum.

    """
    super(NovoGrad2, self).__init__(learning_rate, momentum=beta1,
                                   use_locking=use_locking, name=name,
                                   use_nesterov=False)
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    self._ord = ord

    self._lr = learning_rate
    self._wd = weight_decay
    self._use_locking = use_locking

    # Tensor versions, converted to tensors in apply_gradients
    self._beta1_t = None
    self._beta2_t = None
    self._lr_t =  None
    self._wd_t =  None
    self._grads_ema = None

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    self._beta1_t = ops.convert_to_tensor(self._beta1, name='beta1', dtype = tf.float32)
    self._beta2_t = ops.convert_to_tensor(self._beta2, name='beta2', dtype = tf.float32)
    self._wd_t = ops.convert_to_tensor(self._wd, name='wd', dtype = tf.float32)
    self._lr_t = ops.convert_to_tensor(self._lr, name='lr', dtype = tf.float32)

    len_vars = len(grads_and_vars)
    # init ema variables if required
    if self._grads_ema is None:
      self._grads_ema = [None] * len_vars
      for i in range(len_vars):
        self._grads_ema[i] = tf.get_variable(name="novograd2_ema" + str(i),
                                             shape=[], dtype=tf.float32,
                                             initializer=tf.keras.initializers.Zeros(),
                                             trainable=False)
    # compute ema for grads^2 for each layer
    for i, (grad, var) in enumerate(grads_and_vars):
      # g_norm = tf.norm(tensor=tf.cast(grad, tf.float32), ord=self._ord)
      g_norm = tf.reduce_sum(tf.square(x=tf.cast(grad, tf.float32)))
      self._grads_ema[i] = tf.cond(tf.equal(self._grads_ema[i], 0.),
            lambda: g_norm,
            lambda: self._grads_ema[i]*self._beta2_t + g_norm*(1.-self._beta2_t)
                                   )
      # rescale grads                              )
      g_factor = self._beta1_t / tf.sqrt(self._grads_ema[i] + self._epsilon)
      grad = tf.scalar_mul(g_factor, grad)

      if (self._wd > 0.):
        # var.assign_sub((2*self._wd_t * self._lr_t)*var, self._use_locking)
        var.assign_sub((2*self._wd_t)*var, self._use_locking)

      grads_and_vars[i] = (grad, var)

    return super(NovoGrad2, self).apply_gradients(
      grads_and_vars, global_step=global_step, name=name)


#------------------------------------------------------------------------------
class SethOptimizer(MomentumOptimizer):
  """Optimizer that implements the SAG with layerwise normalized grads

    ```
    m_t <- beta * m_{t-1} + (1 - beta) * (g_t/|g_t|)
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

