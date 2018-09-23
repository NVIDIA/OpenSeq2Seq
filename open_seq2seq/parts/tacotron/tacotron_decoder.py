# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""
Modified by blisc to enable support for tacotron models, specfically enables
the prenet
"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import collections

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest

class BasicDecoderOutput(
    collections.namedtuple(
        "BasicDecoderOutput", ("rnn_output", "stop_token_output")
    )
):
  pass


class TacotronDecoder(decoder.Decoder):
  """Basic sampling decoder."""

  def __init__(
      self,
      decoder_cell,
      helper,
      initial_decoder_state,
      attention_type,
      spec_layer,
      stop_token_layer,
      prenet=None,
      dtype=dtypes.float32,
      train=True
  ):
    """Initialize TacotronDecoder.

    Args:
      decoder_cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_decoder_state: A (possibly nested tuple of...) tensors and
        TensorArrays. The initial state of the RNNCell.
      attention_type: The type of attention used
      stop_token_layer: An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`. Stop token layer to apply to the RNN output to
        predict when to stop the decoder
      spec_layer: An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`. Output layer to apply to the RNN output to map
        the ressult to a spectrogram
      prenet: The prenet to apply to inputs

    Raises:
      TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
    """
    rnn_cell_impl.assert_like_rnncell("cell", decoder_cell)
    if not isinstance(helper, helper_py.Helper):
      raise TypeError("helper must be a Helper, received: %s" % type(helper))
    if (
        spec_layer is not None and
        not isinstance(spec_layer, layers_base.Layer)
    ):
      raise TypeError(
          "spec_layer must be a Layer, received: %s" % type(spec_layer)
      )
    self._decoder_cell = decoder_cell
    self._helper = helper
    self._decoder_initial_state = initial_decoder_state
    self._spec_layer = spec_layer
    self._stop_token_layer = stop_token_layer
    self._attention_type = attention_type
    self._dtype = dtype
    self._prenet = prenet

    if train:
      self._spec_layer = None
      self._stop_token_layer = None

  @property
  def batch_size(self):
    return self._helper.batch_size

  def _rnn_output_size(self):
    size = self._decoder_cell.output_size
    if self._spec_layer is None:
      return size

    output_shape_with_unknown_batch = nest.map_structure(
        lambda s: tensor_shape.TensorShape([None]).concatenate(s), size
    )
    layer_output_shape = self._spec_layer.compute_output_shape(
        output_shape_with_unknown_batch
    )
    return nest.map_structure(lambda s: s[1:], layer_output_shape)

  def _stop_token_output_size(self):
    size = self._decoder_cell.output_size
    if self._stop_token_layer is None:
      return size

    output_shape_with_unknown_batch = nest.map_structure(
        lambda s: tensor_shape.TensorShape([None]).concatenate(s), size
    )
    layer_output_shape = self._stop_token_layer.compute_output_shape(
        output_shape_with_unknown_batch
    )
    return nest.map_structure(lambda s: s[1:], layer_output_shape)

  @property
  def output_size(self):
    return BasicDecoderOutput(
        rnn_output=self._rnn_output_size(),
        stop_token_output=self._stop_token_output_size(),
    )

  @property
  def output_dtype(self):
    # dtype = nest.flatten(self._decoder_initial_state)[0].dtype
    return BasicDecoderOutput(
        nest.map_structure(lambda _: self._dtype, self._rnn_output_size()),
        nest.map_structure(lambda _: self._dtype, self._stop_token_output_size()),
    )

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.
    """
    state = (self._decoder_initial_state, )
    return self._helper.initialize() + state

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
      if self._prenet is not None:
        inputs = self._prenet(inputs)

      cell_outputs, cell_state = self._decoder_cell(inputs, state)

      # If we are training and not using scheduled sampling, we can move
      # all projection layers outside decoder,
      # else we must project inside decoder
      if self._spec_layer is not None:
        spec_outputs = self._spec_layer(cell_outputs)
      else:
        spec_outputs = cell_outputs
      if self._stop_token_layer is not None:
        stop_token_output = self._stop_token_layer(spec_outputs)
      else:
        stop_token_output = cell_outputs

      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=spec_outputs,
          state=cell_state,
          stop_token_predictions=stop_token_output
      )
    outputs = BasicDecoderOutput(spec_outputs, stop_token_output)
    return (outputs, next_state, next_inputs, finished)
