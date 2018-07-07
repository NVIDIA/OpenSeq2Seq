# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import collections

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops


class BasicDecoderOutput(
    collections.namedtuple("BasicDecoderOutput", ("rnn_output", "target_output", "sample_id"))):
  pass


class TacotronDecoder(decoder.Decoder):
  """Basic sampling decoder."""

  def __init__(self, decoder_cell, attention_cell, helper,
              initial_decoder_state, initial_attention_state, 
              attention_type, spec_layer, target_layer,
              use_prenet_output=True,
              stop_token_full=True, prenet=None):
    """Initialize BasicDecoder.
    Args:
      decoder_cell: An `RNNCell` instance.
      attention_cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_decoder_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      initial_attention_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      attention_type: The type of attention used
      target_layer: An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`. Stop token layer to apply to the RNN output to
        predict when to stop the decoder
      spec_layer: An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`. Output layer to apply to the RNN output to map
        the ressult to a spectrogram
      use_prenet_output: (Optional), whether to use the prenet output
        in the attention mechanism
    Raises:
      TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
    """
    rnn_cell_impl.assert_like_rnncell("cell", decoder_cell)
    rnn_cell_impl.assert_like_rnncell("cell", attention_cell)
    if not isinstance(helper, helper_py.Helper):
      raise TypeError("helper must be a Helper, received: %s" % type(helper))
    if (spec_layer is not None
        and not isinstance(spec_layer, layers_base.Layer)):
      raise TypeError(
          "spec_layer must be a Layer, received: %s" % type(spec_layer))
    self.decoder_cell = decoder_cell
    self.attention_cell = attention_cell
    self.helper = helper
    self.decoder_initial_state = initial_decoder_state
    self.attention_initial_state = initial_attention_state
    self.spec_layer = spec_layer
    self.target_layer = target_layer
    self.attention_type = attention_type
    self.use_prenet_output = use_prenet_output
    self._dtype = initial_decoder_state[0].h.dtype
    self.stop_token_full = stop_token_full
    self.prenet = prenet

  @property
  def batch_size(self):
    return self.helper.batch_size

  def _rnn_output_size(self):
    size = self.decoder_cell.output_size
    if self.spec_layer is None:
      return size
    else:
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = nest.map_structure(
          lambda s: tensor_shape.TensorShape([None]).concatenate(s),
          size)
      layer_output_shape = self.spec_layer.compute_output_shape(
          output_shape_with_unknown_batch)
      return nest.map_structure(lambda s: s[1:], layer_output_shape)

  def _stop_token_output_size(self):
    size = self.decoder_cell.output_size
    if self.target_layer is None:
      return size
    else:
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = nest.map_structure(
          lambda s: tensor_shape.TensorShape([None]).concatenate(s),
          size)
      layer_output_shape = self.target_layer.compute_output_shape(
          output_shape_with_unknown_batch)
      return nest.map_structure(lambda s: s[1:], layer_output_shape)

  @property
  def output_size(self):
    # Return the cell output and the id
    return BasicDecoderOutput(
        rnn_output=self._rnn_output_size(),
        target_output=self._stop_token_output_size(),
        sample_id=self.helper.sample_ids_shape)

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and the sample_ids_dtype from the helper.
    dtype = nest.flatten(self.decoder_initial_state)[0].dtype
    return BasicDecoderOutput(
        nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        nest.map_structure(lambda _: dtype, self._stop_token_output_size()),
        self.helper.sample_ids_dtype)

  def initialize(self, name=None):
    """Initialize the decoder.
    Args:
      name: Name scope for any created operations.
    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    if self.attention_type == "location":
      self.attention_cell._attention_mechanisms[0].initialize_location(self._dtype)
    return self.helper.initialize() + ((self.attention_initial_state,self.decoder_initial_state,),)

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
      if self.prenet is not None:
        inputs = self.prenet(inputs)

      if self.use_prenet_output:
        attention_context, attention_state = self.attention_cell(inputs, state[0])
      else:
        # Do an attention rnn step with the decoder output and previous attention state
        attention_context, attention_state = self.attention_cell(state[1].h, state[0])
      # For the decoder rnn, the input is the (prenet) output + attention context + attention rnn state
      decoder_rnn_input = array_ops.concat((inputs,attention_context), axis=-1)
      # decoder_rnn_input = state[0].cell_state.h
      # decoder_rnn_input = array_ops.concat((inputs,attention_context, state[0].cell_state.h), axis=-1)
      cell_outputs, decoder_state = self.decoder_cell(decoder_rnn_input, state[1])
      cell_state = (attention_state, decoder_state)
      # Concatenate the decoder output and attention output and send it through a projection layer
      cell_outputs = array_ops.concat((cell_outputs, attention_context), axis=-1)
      spec_outputs = self.spec_layer(cell_outputs)
      # test removing the cell outputs fix
      if self.stop_token_full:
        target_outputs = self.target_layer(spec_outputs)
      else:
        target_outputs = self.target_layer(cell_outputs)
      sample_ids = self.helper.sample(
          time=time, outputs=spec_outputs, state=cell_state)
      (finished, next_inputs, next_state) = self.helper.next_inputs(
          time=time,
          outputs=spec_outputs,
          state=cell_state,
          sample_ids=sample_ids,
          stop_token_predictions=target_outputs)
    outputs = BasicDecoderOutput(spec_outputs, target_outputs, sample_ids)
    return (outputs, next_state, next_inputs, finished)