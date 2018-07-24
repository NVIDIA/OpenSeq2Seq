from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from six.moves import range

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.nn_ops import dropout


class ZoneoutWrapper(rnn_cell_impl.RNNCell):
  """Operator adding zoneout to all states (states+cells) of the given cell.
  Code taken from https://github.com/teganmaharaj/zoneout
  applying zoneout as described in https://arxiv.org/pdf/1606.01305.pdf"""

  def __init__(self, cell, zoneout_prob, is_training=True, seed=None):
    if not isinstance(cell, rnn_cell_impl.RNNCell):
      raise TypeError("The parameter cell is not an RNNCell.")
    if (
        isinstance(zoneout_prob, float) and
        not (zoneout_prob >= 0.0 and zoneout_prob <= 1.0)
    ):
      raise ValueError(
          "Parameter zoneout_prob must be between 0 and 1: %d" % zoneout_prob
      )
    self._cell = cell
    self._zoneout_prob = (zoneout_prob, zoneout_prob)
    self._seed = seed
    self._is_training = is_training

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    if isinstance(self.state_size,
                  tuple) != isinstance(self._zoneout_prob, tuple):
      raise TypeError("Subdivided states need subdivided zoneouts.")
    if isinstance(self.state_size,
                  tuple) and len(tuple(self.state_size)
                                ) != len(tuple(self._zoneout_prob)):
      raise ValueError("State and zoneout need equally many parts.")
    output, new_state = self._cell(inputs, state, scope)
    if isinstance(self.state_size, tuple):
      if self._is_training:
        new_state = tuple(
            (1 - state_part_zoneout_prob) * dropout(
                new_state_part - state_part, (1 - state_part_zoneout_prob),
                seed=self._seed
            ) + state_part
            for new_state_part, state_part, state_part_zoneout_prob in
            zip(new_state, state, self._zoneout_prob)
        )
      else:
        new_state = tuple(
            state_part_zoneout_prob * state_part +
            (1 - state_part_zoneout_prob) * new_state_part
            for new_state_part, state_part, state_part_zoneout_prob in
            zip(new_state, state, self._zoneout_prob)
        )
      new_state = rnn_cell_impl.LSTMStateTuple(new_state[0], new_state[1])
    else:
      raise ValueError("Only states that are tuples are supported")
    return output, new_state
