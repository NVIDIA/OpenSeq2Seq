# Copyright (c) 2019 NVIDIA Corporation
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
  Optimizer that implements SGD with layer-wise normalized gradients,
  when normalization is done by sqrt(ema(sqr(grads))), similar to Adam

    ```
    Second moment = ema of Layer-wise sqr of grads:
       v_t <-- beta2*v_{t-1} + (1-beta2)*(g_t)^2

    First moment has two mode:
    1. moment of grads normalized by u_t:
       m_t <- beta1*m_{t-1} + lr_t * [ g_t/sqrt(v_t+epsilon)]
    1. moment similar to Adam: ema of grads normalized by u_t:
       m_t <- beta1*m_{t-1} + lr_t * [(1-beta1)*(g_t/sqrt(v_t+epsilon))]

    if weight decay add wd term after grads are rescaled by 1/sqrt(v_t):
       m_t <- beta1*m_{t-1} + lr_t * [g_t/sqrt(v_t+epsilon) + wd*w_{t-1}]

    Weight update:
       w_t <- w_{t-1} - *m_t
    ```

  """

  def __init__(self,
               learning_rate=1.0,
               beta1=0.95,
               beta2=0.98,
               epsilon=1e-8,
               weight_decay=0.0,
               grad_averaging=False,
               use_locking=False,
               name='NovoGrad'):
    """Constructor:

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      beta1: A `Tensor` or a float, used in ema for momentum.Default = 0.95.
      beta2: A `Tensor` or a float, used in ema for grad norms.Default = 0.99.
      epsilon: a float.  Default = 1e-8.
      weight_decay: A `Tensor` or a float, Default = 0.0.
      grad_averaging: switch between Momentum and SAG, Default = False,
      use_locking: If `True` use locks for update operations.
      name: Optional, name prefix for the ops created when applying
        gradients.  Defaults to "NovoGrad".
      use_nesterov: If `True` use Nesterov Momentum.

    """
    super(NovoGrad, self).__init__(learning_rate, momentum=beta1,
                                   use_locking=use_locking, name=name,
                                   use_nesterov=False)
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    self._wd  = weight_decay
    self._grad_averaging  = grad_averaging
    self._grads_ema = None

    # Tensor versions, converted to tensors in apply_gradients
    # self._beta1_t = None
    # self._beta2_t = None
    # self._wd_t = None

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    # self._beta1_t = ops.convert_to_tensor(self._beta1, name='beta1', dtype = tf.float32)
    # self._beta2_t = ops.convert_to_tensor(self._beta2, name='beta2', dtype = tf.float32)

    # init ema variables if required
    len_vars = len(grads_and_vars)
    if self._grads_ema is None:
      self._grads_ema = [None] * len_vars
      for i in range(len_vars):
        self._grads_ema[i] = tf.get_variable(name="nvgrad2_ema" + str(i),
                                     shape=[], dtype=tf.float32,
                                     initializer=tf.keras.initializers.Zeros(),
                                     trainable=False)

    # compute ema for grads^2 for each layer
    for i, (grad, var) in enumerate(grads_and_vars):
      g_2 = tf.reduce_sum(tf.square(x=tf.cast(grad, tf.float32)))
      self._grads_ema[i] = tf.cond(tf.equal(self._grads_ema[i], 0.),
                  lambda: g_2,
                  lambda: self._grads_ema[i]*self._beta2 + g_2*(1.-self._beta2)
                  )

      grad *= 1.0 / tf.sqrt(self._grads_ema[i] + self._epsilon)
      # weight decay
      if (self._wd > 0.):
        grad += (self._wd * var)
      # Momentum --> SAG
      if self._grad_averaging:
        grad *= (1.-self._beta1)
      grads_and_vars[i] = (grad, var)

    # call Momentum to do update
    return super(NovoGrad, self).apply_gradients(
         grads_and_vars, global_step=global_step, name=name)

