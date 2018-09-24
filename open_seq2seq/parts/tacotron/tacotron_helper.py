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
Modified by blisc to enable support for tacotron models
Custom Helper class that implements the tacotron decoder pre and post nets
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops.helper import Helper
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest

_transpose_batch_time = decoder._transpose_batch_time


def _unstack_ta(inp):
  return tensor_array_ops.TensorArray(
      dtype=inp.dtype,
      size=array_ops.shape(inp)[0],
      element_shape=inp.get_shape()[1:]
  ).unstack(inp)


class TacotronTrainingHelper(Helper):
  """Helper funciton for training. Can be used for teacher forcing or scheduled
  sampling"""

  def __init__(
      self,
      inputs,
      sequence_length,
      prenet=None,
      time_major=False,
      sample_ids_shape=None,
      sample_ids_dtype=None,
      model_dtype=tf.float32,
      mask_decoder_sequence=None
  ):
    """Initializer.

    Args:
      inputs (Tensor): inputs of shape [batch, time, n_feats]
      sequence_length (Tensor): length of each input. shape [batch]
      prenet: prenet to use, currently disabled and used in tacotron decoder
        instead.
      sampling_prob (float): see tacotron 2 decoder
      time_major (bool): (float): see tacotron 2 decoder
      mask_decoder_sequence (bool): whether to pass finished when the decoder
        passed the sequence_length input or to pass unfinished to dynamic_decode
    """
    self._sample_ids_shape = tensor_shape.TensorShape(sample_ids_shape or [])
    self._sample_ids_dtype = sample_ids_dtype or dtypes.int32

    if not time_major:
      inputs = nest.map_structure(_transpose_batch_time, inputs)
    self._input_tas = nest.map_structure(_unstack_ta, inputs)
    self._sequence_length = sequence_length
    self._batch_size = array_ops.size(sequence_length)
    self._seed = None
    self._mask_decoder_sequence = mask_decoder_sequence
    self._prenet = prenet
    self._zero_inputs = nest.map_structure(
        lambda inp: array_ops.zeros_like(inp[0, :]), inputs
    )
    self._start_inputs = self._zero_inputs
    if prenet is not None:
      self._start_inputs = self._prenet(self._zero_inputs)
    self._last_dim = self._start_inputs.get_shape()[-1]
    self._dtype = model_dtype

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return self._sample_ids_shape

  @property
  def sample_ids_dtype(self):
    return self._sample_ids_dtype

  def initialize(self, name=None):
    finished = array_ops.tile([False], [self._batch_size])
    return (finished, self._start_inputs)

  def sample(self, time, outputs, state, name=None):
    # Fully deterministic, output should already be projected
    pass

  def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
    # Applies the fully connected pre-net to the decoder
    # Also decides whether the decoder is finished
    next_time = time + 1
    if self._mask_decoder_sequence:
      finished = (next_time >= self._sequence_length)
    else:
      finished = array_ops.tile([False], [self._batch_size])
    all_finished = math_ops.reduce_all(finished)

    def get_next_input(inp, out):
      next_input = inp.read(time)
      if self._prenet is not None:
        next_input = self._prenet(next_input)
        out = self._prenet(out)
      return next_input

    next_inputs = control_flow_ops.cond(
        all_finished, lambda: self._start_inputs,
        lambda: get_next_input(self._input_tas, outputs)
    )

    return (finished, next_inputs, state)


class TacotronHelper(Helper):
  """Helper for use during eval and infer. Does not use teacher forcing"""

  def __init__(
      self,
      inputs,
      prenet=None,
      time_major=False,
      sample_ids_shape=None,
      sample_ids_dtype=None,
      mask_decoder_sequence=None
  ):
    """Initializer.

    Args:
      inputs (Tensor): inputs of shape [batch, time, n_feats]
      prenet: prenet to use, currently disabled and used in tacotron decoder
        instead.
      sampling_prob (float): see tacotron 2 decoder
      anneal_teacher_forcing (float): see tacotron 2 decoder
      stop_gradient (float): see tacotron 2 decoder
      time_major (bool): (float): see tacotron 2 decoder
      mask_decoder_sequence (bool): whether to pass finished when the decoder
        passed the sequence_length input or to pass unfinished to dynamic_decode
    """
    self._sample_ids_shape = tensor_shape.TensorShape(sample_ids_shape or [])
    self._sample_ids_dtype = sample_ids_dtype or dtypes.int32
    self._batch_size = inputs.get_shape()[0]
    self._mask_decoder_sequence = mask_decoder_sequence

    if not time_major:
      inputs = nest.map_structure(_transpose_batch_time, inputs)

    inputs = inputs[0, :, :]
    self._prenet = prenet
    if prenet is None:
      self._start_inputs = inputs
    else:
      self._start_inputs = self._prenet(inputs)

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return self._sample_ids_shape

  @property
  def sample_ids_dtype(self):
    return self._sample_ids_dtype

  def initialize(self, name=None):
    finished = array_ops.tile([False], [self._batch_size])
    return (finished, self._start_inputs)

  def sample(self, time, outputs, state, name=None):
    # Fully deterministic, output should already be projected
    pass
  def next_inputs(
      self,
      time,
      outputs,
      state,
      stop_token_predictions,
      name=None,
      **unused_kwargs
  ):
    # Applies the fully connected pre-net to the decoder
    # Also decides whether the decoder is finished
    next_time = time + 1
    if self._mask_decoder_sequence:
      stop_token_predictions = tf.sigmoid(stop_token_predictions)
      finished = tf.cast(tf.round(stop_token_predictions), tf.bool)
      finished = tf.squeeze(finished)
    else:
      finished = array_ops.tile([False], [self._batch_size])
    all_finished = math_ops.reduce_all(finished)

    def get_next_input(out):
      if self._prenet is not None:
        out = self._prenet(out)
      return out

    next_inputs = control_flow_ops.cond(
        all_finished, lambda: self._start_inputs,
        lambda: get_next_input(outputs)
    )
    return (finished, next_inputs, state)
