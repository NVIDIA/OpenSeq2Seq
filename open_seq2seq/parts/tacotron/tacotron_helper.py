# Copyright (c) 2018 NVIDIA Corporation
"""
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
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops import gen_array_ops

_transpose_batch_time = decoder._transpose_batch_time

def _unstack_ta(inp):
  return tensor_array_ops.TensorArray(
      dtype=inp.dtype, size=array_ops.shape(inp)[0],
      element_shape=inp.get_shape()[1:]).unstack(inp)

class TacotronTrainingHelper(Helper):
  """Helper funciton for training. Can be used for teacher forcing or scheduled sampling"""

  def __init__(self, inputs, sequence_length, enable_prenet, 
              prenet_units=None, prenet_layers=None, prenet_activation=None,
              sampling_prob=0., anneal_sampling_prob = False, sampling_test=False,
              time_major=False, sample_ids_shape=None, sample_ids_dtype=None, name=None,
              context=None, mask_decoder_sequence=None):
    """Initializer.
    Args:
      To-Do
    """
    self._sample_ids_shape = tensor_shape.TensorShape(sample_ids_shape or [])
    self._sample_ids_dtype = sample_ids_dtype or dtypes.int32

    if not time_major:
      inputs = nest.map_structure(_transpose_batch_time, inputs)
    self._input_tas = nest.map_structure(_unstack_ta, inputs)
    self._sequence_length = sequence_length
    self._zero_inputs = nest.map_structure(
        lambda inp: array_ops.zeros_like(inp[0, :]), inputs)
    self._batch_size = array_ops.size(sequence_length)
    self.seed = 0
    self.anneal_sampling_prob = anneal_sampling_prob
    self.sampling_prob = sampling_prob
    self.sampling_test = sampling_test
    self.mask_decoder_sequence = mask_decoder_sequence
    # self.context = context

    ## Create dense pre_net
    self.prenet_layers=[]
    if enable_prenet:
      for idx in range(prenet_layers):
        self.prenet_layers.append(tf.layers.Dense(
          name="prenet_{}".format(idx + 1),
          units=prenet_units,
          activation=prenet_activation,
          use_bias=False,
        ))

      pre_net_output = tf.zeros([self._batch_size, prenet_units])
      self._zero_inputs = pre_net_output
    else:
      self._zero_inputs = nest.map_structure(
        lambda inp: array_ops.zeros_like(inp[0, :]), inputs)
    self.last_dim = self._zero_inputs.get_shape()[-1]
    # self._zero_inputs = tf.concat([self._zero_inputs, self.context],axis=-1)
    # self._zero_inputs = tf.concat([pre_net_output,self._zero_inputs], axis=-1)

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
    return (finished, self._zero_inputs )

  def sample(self, time, outputs, state, name=None):
    # Fully deterministic, output should already be projected
    del time, state
    sample_ids = math_ops.cast(
            math_ops.argmax(outputs, axis=-1), dtypes.int32)
    return sample_ids

  def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
    # Applies the fully connected pre-net to the decoder
    # Also decides whether the decoder is finished
    next_time = time + 1
    if self.mask_decoder_sequence:
      finished = (next_time >= self._sequence_length)
    else:
      finished = array_ops.tile([False], [self._batch_size])
    all_finished = math_ops.reduce_all(finished)
    def get_next_input(inp, out):
      next_input = inp.read(time)
      if self.sampling_test:
        next_input = tf.stop_gradient(next_input)
        out = tf.stop_gradient(out)
      for layer in self.prenet_layers:
        next_input = tf.layers.dropout(layer(next_input), rate=0.5, training=True)
        out = tf.layers.dropout(layer(out), rate=0.5, training=True)
      if self.anneal_sampling_prob or self.sampling_prob > 0:
        select_sampler = bernoulli.Bernoulli(
            probs=self.sampling_prob, dtype=dtypes.bool)
        select_sample = select_sampler.sample(
            sample_shape=(self.batch_size,1), seed=self.seed)
        select_sample = tf.tile(select_sample, [1,self.last_dim])
        sample_ids = array_ops.where(
            select_sample,
            out,
            gen_array_ops.fill([self.batch_size, self.last_dim], -20.))
        where_sampling = math_ops.cast(
            array_ops.where(sample_ids > -20), dtypes.int32)
        where_not_sampling = math_ops.cast(
            array_ops.where(sample_ids <= -20), dtypes.int32)
        sample_ids_sampling = array_ops.gather_nd(sample_ids, where_sampling)
        inputs_not_sampling = array_ops.gather_nd(
            next_input, where_not_sampling)
        sampled_next_inputs = sample_ids_sampling
        base_shape = array_ops.shape(next_input)

        next_input = (array_ops.scatter_nd(indices=where_sampling,
                                       updates=sampled_next_inputs,
                                       shape=base_shape)
                  + array_ops.scatter_nd(indices=where_not_sampling,
                                         updates=inputs_not_sampling,
                                         shape=base_shape))
      return next_input

    next_inputs = control_flow_ops.cond(
          all_finished, 
          lambda: self._zero_inputs,
          lambda: get_next_input(self._input_tas, outputs))

    return (finished, next_inputs, state)


class TacotronHelper(Helper):
  """Helper for use during eval and infer. Does not use teacher forcing"""

  def __init__(self, inputs, enable_prenet=True, 
              prenet_units=None, prenet_layers=None, prenet_activation=None,
              time_major=False, sample_ids_shape=None, sample_ids_dtype=None, name=None,
              context=None, mask_decoder_sequence=None):
    """Initializer.
    Args:
      To-Do
    """
    self._sample_ids_shape = tensor_shape.TensorShape(sample_ids_shape or [])
    self._sample_ids_dtype = sample_ids_dtype or dtypes.int32

    self._batch_size = inputs.get_shape()[0]
    self.mask_decoder_sequence = mask_decoder_sequence
    # self.context = context

    self.prenet_layers=[]
    if enable_prenet:
      for idx in range(prenet_layers):
        self.prenet_layers.append(tf.layers.Dense(
          name="prenet_{}".format(idx + 1),
          units=prenet_units,
          activation=prenet_activation,
          use_bias=False,
        ))

      pre_net_output = tf.zeros([self._batch_size, prenet_units])
      self._zero_inputs = pre_net_output
    else:
      self._zero_inputs = inputs
    # self._zero_inputs = tf.concat([self._zero_inputs, self.context],axis=-1)

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
    return (finished, self._zero_inputs)

  def sample(self, time, outputs, state, name=None):
    # Fully deterministic, output should already be projected
    del time, state
    sample_ids = math_ops.cast(
            math_ops.argmax(outputs, axis=-1), dtypes.int32)
    return sample_ids

  def next_inputs(self, time, outputs, state, stop_token_predictions, name=None, **unused_kwargs):
    # Applies the fully connected pre-net to the decoder
    # Also decides whether the decoder is finished
    next_time = time + 1
    if self.mask_decoder_sequence:
      stop_token_predictions = tf.sigmoid(stop_token_predictions)
      finished = tf.cast(tf.round(stop_token_predictions), tf.bool)
      finished = tf.squeeze(finished)
    else:
      finished = array_ops.tile([False], [self._batch_size])
    all_finished = math_ops.reduce_all(finished)

    def get_next_input(out):
      # next_input = tf.concat([pre_net_result, outputs], axis=-1)
      for layer in self.prenet_layers:
        out = tf.layers.dropout(layer(out), rate=0.5, training=True)
      return out
    next_inputs = control_flow_ops.cond(
        all_finished, 
        lambda: self._zero_inputs,
        lambda: get_next_input(outputs))
    return (finished, next_inputs, state)