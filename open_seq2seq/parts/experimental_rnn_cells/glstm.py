"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from six.moves import range

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs

# pylint: disable=protected-access
_Linear = core_rnn_cell._Linear  # pylint: disable=invalid-name
# pylint: enable=protected-access


# TODO: must implement all abstract methods
class GLSTMCell(rnn_cell_impl.RNNCell):
  """Group LSTM cell (G-LSTM).
  The implementation is based on:
    https://arxiv.org/abs/1703.10722
  O. Kuchaiev and B. Ginsburg
  "Factorization Tricks for LSTM Networks", ICLR 2017 workshop.
  """

  def __init__(self, num_units, initializer=None, num_proj=None,
               number_of_groups=1, forget_bias=1.0, activation=math_ops.tanh,
               reuse=None):
    """Initialize the parameters of G-LSTM cell.
    Args:
      num_units: int, The number of units in the G-LSTM cell
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      number_of_groups: (optional) int, number of groups to use.
        If `number_of_groups` is 1, then it should be equivalent to LSTM cell
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training.
      activation: Activation function of the inner states.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already
        has the given variables, an error is raised.
    Raises:
      ValueError: If `num_units` or `num_proj` is not divisible by
        `number_of_groups`.
    """
    super(GLSTMCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._initializer = initializer
    self._num_proj = num_proj
    self._forget_bias = forget_bias
    self._activation = activation
    self._number_of_groups = number_of_groups

    if self._num_units % self._number_of_groups != 0:
      raise ValueError("num_units must be divisible by number_of_groups")
    if self._num_proj:
      if self._num_proj % self._number_of_groups != 0:
        raise ValueError("num_proj must be divisible by number_of_groups")
      self._group_shape = [int(self._num_proj / self._number_of_groups),
                           int(self._num_units / self._number_of_groups)]
    else:
      self._group_shape = [int(self._num_units / self._number_of_groups),
                           int(self._num_units / self._number_of_groups)]

    if num_proj:
      self._state_size = rnn_cell_impl.LSTMStateTuple(num_units, num_proj)
      self._output_size = num_proj
    else:
      self._state_size = rnn_cell_impl.LSTMStateTuple(num_units, num_units)
      self._output_size = num_units
    self._linear1 = [None] * self._number_of_groups
    self._linear2 = None

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def _get_input_for_group(self, inputs, group_id, group_size):
    """Slices inputs into groups to prepare for processing by cell's groups
    Args:
      inputs: cell input or it's previous state,
              a Tensor, 2D, [batch x num_units]
      group_id: group id, a Scalar, for which to prepare input
      group_size: size of the group
    Returns:
      subset of inputs corresponding to group "group_id",
      a Tensor, 2D, [batch x num_units/number_of_groups]
    """
    return array_ops.slice(input_=inputs,
                           begin=[0, group_id * group_size],
                           size=[self._batch_size, group_size],
                           name=("GLSTM_group%d_input_generation" % group_id))

  # TODO: does not match signature of the base method
  def call(self, inputs, state):
    """Run one step of G-LSTM.
    Args:
      inputs: input Tensor, 2D, [batch x num_units].
      state: this must be a tuple of state Tensors, both `2-D`,
      with column sizes `c_state` and `m_state`.
    Returns:
      A tuple containing:
      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        G-LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - LSTMStateTuple representing the new state of G-LSTM cell
        after reading `inputs` when the previous state was `state`.
    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    (c_prev, m_prev) = state

    self._batch_size = inputs.shape[0].value or array_ops.shape(inputs)[0]
    input_size = inputs.shape[-1].value or array_ops.shape(inputs)[-1]
    dtype = inputs.dtype
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope, initializer=self._initializer):
      i_parts = []
      j_parts = []
      f_parts = []
      o_parts = []

      for group_id in range(self._number_of_groups):
        with vs.variable_scope("group%d" % group_id):
          x_g_id = array_ops.concat(
            [self._get_input_for_group(inputs, group_id,
                                       int(input_size / self._number_of_groups)),
                                       # self._group_shape[0]), # this is only correct if inputs dim = num_units!!!
             self._get_input_for_group(m_prev, group_id,
                                       int(self._output_size / self._number_of_groups))], axis=1)
                                       # self._group_shape[0])], axis=1)
          if self._linear1[group_id] is None:
            self._linear1[group_id] = _Linear(
              x_g_id, 4 * self._group_shape[1],
              False,
            )
          R_k = self._linear1[group_id](x_g_id)  # pylint: disable=invalid-name
          i_k, j_k, f_k, o_k = array_ops.split(R_k, 4, 1)

        i_parts.append(i_k)
        j_parts.append(j_k)
        f_parts.append(f_k)
        o_parts.append(o_k)

      bi = vs.get_variable(
        name="bias_i",
        shape=[self._num_units],
        dtype=dtype,
        initializer=init_ops.constant_initializer(0.0, dtype=dtype),
      )
      bj = vs.get_variable(
        name="bias_j",
        shape=[self._num_units],
        dtype=dtype,
        initializer=init_ops.constant_initializer(0.0, dtype=dtype),
      )
      bf = vs.get_variable(
        name="bias_f",
        shape=[self._num_units],
        dtype=dtype,
        initializer=init_ops.constant_initializer(0.0, dtype=dtype),
      )
      bo = vs.get_variable(
        name="bias_o",
        shape=[self._num_units],
        dtype=dtype,
        initializer=init_ops.constant_initializer(0.0, dtype=dtype),
      )

      i = nn_ops.bias_add(array_ops.concat(i_parts, axis=1), bi)
      j = nn_ops.bias_add(array_ops.concat(j_parts, axis=1), bj)
      f = nn_ops.bias_add(array_ops.concat(f_parts, axis=1), bf)
      o = nn_ops.bias_add(array_ops.concat(o_parts, axis=1), bo)

    c = (math_ops.sigmoid(f + self._forget_bias) * c_prev +
         math_ops.sigmoid(i) * math_ops.tanh(j))
    m = math_ops.sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      with vs.variable_scope("projection"):
        if self._linear2 is None:
          self._linear2 = _Linear(m, self._num_proj, False)
        m = self._linear2(m)

    new_state = rnn_cell_impl.LSTMStateTuple(c, m)
    return m, new_state
