# Copyright 2017 Google Inc. All Rights Reserved.
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

# THIS CODE WAS TAKEN FROM:
#   https://raw.githubusercontent.com/tensorflow/nmt/master/nmt/gnmt_model.py

"""GNMT attention sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf

from tensorflow.python.util import nest


# TODO: must implement all abstract methods
class GNMTAttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
  """A MultiCell with GNMT attention style."""

  def __init__(self, attention_cell, cells, use_new_attention=False):
    """Creates a GNMTAttentionMultiCell.

    Args:
      attention_cell: An instance of AttentionWrapper.
      cells: A list of RNNCell wrapped with AttentionInputWrapper.
      use_new_attention: Whether to use the attention generated from current
        step bottom layer's output. Default is False.
    """
    cells = [attention_cell] + cells
    self.use_new_attention = use_new_attention
    super(GNMTAttentionMultiCell, self).__init__(cells, state_is_tuple=True)

  # TODO: does not match signature of the base method
  def __call__(self, inputs, state, scope=None):
    """Run the cell with bottom layer's attention copied to all upper layers."""
    if not nest.is_sequence(state):
      raise ValueError(
          "Expected state to be a tuple of length %d, but received: %s"
          % (len(self.state_size), state))

    with tf.variable_scope(scope or "multi_rnn_cell"):
      new_states = []

      with tf.variable_scope("cell_0_attention"):
        attention_cell = self._cells[0]
        attention_state = state[0]
        cur_inp, new_attention_state = attention_cell(inputs, attention_state)
        new_states.append(new_attention_state)

      for i in range(1, len(self._cells)):
        with tf.variable_scope("cell_%d" % i):

          cell = self._cells[i]
          cur_state = state[i]

          if self.use_new_attention:
            cur_inp = tf.concat([cur_inp, new_attention_state.attention], -1)
          else:
            cur_inp = tf.concat([cur_inp, attention_state.attention], -1)

          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)

    return cur_inp, tuple(new_states)


def gnmt_residual_fn(inputs, outputs):
  """Residual function that handles different inputs and outputs inner dims.

  Args:
    inputs: cell inputs, this is actual inputs concatenated with the attention
      vector.
    outputs: cell outputs

  Returns:
    outputs + actual inputs
  """
  def split_input(inp, out):
    out_dim = out.get_shape().as_list()[-1]
    inp_dim = inp.get_shape().as_list()[-1]
    return tf.split(inp, [out_dim, inp_dim - out_dim], axis=-1)

  actual_inputs, _ = nest.map_structure(split_input, inputs, outputs)

  def assert_shape_match(inp, out):
    inp.get_shape().assert_is_compatible_with(out.get_shape())

  nest.assert_same_structure(actual_inputs, outputs)
  nest.map_structure(assert_shape_match, actual_inputs, outputs)
  return nest.map_structure(lambda inp, out: inp + out, actual_inputs, outputs)
