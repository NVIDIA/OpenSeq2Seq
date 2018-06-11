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

# class TacotronTrainingHelper(Helper):
#   """Base abstract class that allows the user to customize sampling."""

#   def __init__(self, inputs, sequence_length, time_major=False, sample_ids_shape=None, sample_ids_dtype=None, name=None):
#     """Initializer.
#     Args:
#       initialize_fn: callable that returns `(finished, next_inputs)`
#         for the first iteration.
#       sample_fn: callable that takes `(time, outputs, state)`
#         and emits tensor `sample_ids`.
#       next_inputs_fn: callable that takes `(time, outputs, state, sample_ids)`
#         and emits `(finished, next_inputs, next_state)`.
#       sample_ids_shape: Either a list of integers, or a 1-D Tensor of type
#         `int32`, the shape of each value in the `sample_ids` batch. Defaults to
#         a scalar.
#       sample_ids_dtype: The dtype of the `sample_ids` tensor. Defaults to int32.
#     """
#     self._sample_ids_shape = tensor_shape.TensorShape(sample_ids_shape or [])
#     self._sample_ids_dtype = sample_ids_dtype or dtypes.int32

#     if not time_major:
#       inputs = nest.map_structure(_transpose_batch_time, inputs)
#     self._input_tas = nest.map_structure(_unstack_ta, inputs)
#     self._sequence_length = sequence_length
#     self._zero_inputs = nest.map_structure(
#         lambda inp: array_ops.zeros_like(inp[0, :]), inputs)
#     self._batch_size = array_ops.size(sequence_length)

#     ## Create finished projection layer
#     self._finished_proj_layer = tf.layers.Dense(
#       units=1,
#       activation=tf.nn.sigmoid,
#       use_bias=False,
#     )
#     ## Create dense pre_net
#     self._pre_net_layer_1 = tf.layers.Dense(
#       units=256,
#       activation=tf.nn.relu,
#       use_bias=True,
#     )
#     self._pre_net_layer_2 = tf.layers.Dense(
#       units=256,
#       activation=tf.nn.relu,
#       use_bias=True,
#     )

#     pre_net_output = tf.zeros([self._batch_size, 256])
#     self._zero_inputs = pre_net_output
#     # self._zero_inputs = tf.concat([pre_net_output,self._zero_inputs], axis=-1)

#   @property
#   def batch_size(self):
#     return self._batch_size

#   @property
#   def sample_ids_shape(self):
#     return self._sample_ids_shape

#   @property
#   def sample_ids_dtype(self):
#     return self._sample_ids_dtype

#   def initialize(self, name=None):
#     finished = array_ops.tile([False], [self._batch_size])
#     # next_inputs = self._input_tas.read(0)
#     return (finished, self._zero_inputs)

#   def sample(self, time, outputs, state, name=None):
#     # Fully deterministic, output should already be projected
#     del time, state
#     # Required to appease tensorflow for some reason
#     sample_ids = math_ops.cast(
#             math_ops.argmax(outputs, axis=-1), dtypes.int32)
#     return sample_ids

#   def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
#     # Applies the fully connected pre-net to the decoder
#     # Also decides whether the decoder is finished
#     next_time = time + 1
#     finished = (next_time >= self._sequence_length)
#     all_finished = math_ops.reduce_all(finished)
#     def get_next_input(inp, out):
#       inp= inp.read(time)
#       # pre_net_result = self._pre_net_layer_2(self._pre_net_layer_1(outputs))
#       # next_input = tf.concat([pre_net_result, inp], axis=-1)
#       next_input = self._pre_net_layer_2(self._pre_net_layer_1(inp))
#       return next_input
#     # next_input =  nest.map_structure(read_from_ta, self._input_tas)
#     next_inputs = control_flow_ops.cond(
#         all_finished, 
#         lambda: self._zero_inputs,
#         lambda: get_next_input(self._input_tas, outputs))

#     # print(next_input.shape)
#     # print(next_inputs.shape)
#     # input()

#     return (finished, next_inputs, state)


# class TacotronHelper(Helper):
#   """Base abstract class that allows the user to customize sampling."""

#   def __init__(self, inputs, sequence_length, time_major=False, sample_ids_shape=None, sample_ids_dtype=None):
#     """Initializer.
#     Args:
#       initialize_fn: callable that returns `(finished, next_inputs)`
#         for the first iteration.
#       sample_fn: callable that takes `(time, outputs, state)`
#         and emits tensor `sample_ids`.
#       next_inputs_fn: callable that takes `(time, outputs, state, sample_ids)`
#         and emits `(finished, next_inputs, next_state)`.
#       sample_ids_shape: Either a list of integers, or a 1-D Tensor of type
#         `int32`, the shape of each value in the `sample_ids` batch. Defaults to
#         a scalar.
#       sample_ids_dtype: The dtype of the `sample_ids` tensor. Defaults to int32.
#     """
#     self._sample_ids_shape = tensor_shape.TensorShape(sample_ids_shape or [])
#     self._sample_ids_dtype = sample_ids_dtype or dtypes.int32

#     if not time_major:
#       inputs = nest.map_structure(_transpose_batch_time, inputs)
#     self._input_tas = nest.map_structure(_unstack_ta, inputs)
#     self._sequence_length = sequence_length
#     self._zero_inputs = nest.map_structure(
#         lambda inp: array_ops.zeros_like(inp[0, :]), inputs)
#     self._batch_size = array_ops.size(sequence_length)

#     ## Create finished projection layer
#     self._finished_proj_layer = tf.layers.Dense(
#       units=1,
#       activation=tf.nn.sigmoid,
#       use_bias=False,
#     )
#     ## Create dense pre_net
#     self._pre_net_layer_1 = tf.layers.Dense(
#       units=256,
#       activation=tf.nn.relu,
#       use_bias=True,
#     )
#     self._pre_net_layer_2 = tf.layers.Dense(
#       units=256,
#       activation=tf.nn.relu,
#       use_bias=True,
#     )

#     pre_net_output = tf.zeros([self._batch_size, 256])
#     # self._zero_inputs = tf.concat([pre_net_output,self._zero_inputs], axis=-1)
#     self._zero_inputs = pre_net_output

#   @property
#   def batch_size(self):
#     return self._batch_size

#   @property
#   def sample_ids_shape(self):
#     return self._sample_ids_shape

#   @property
#   def sample_ids_dtype(self):
#     return self._sample_ids_dtype

#   def initialize(self, name=None):
#     finished = array_ops.tile([False], [self._batch_size])
#     # next_inputs = self._input_tas.read(0)
#     return (finished, self._zero_inputs)

#   def sample(self, time, outputs, state, name=None):
#     # Fully deterministic, output should already be projected
#     del time, state
#     # Required to appease tensorflow for some reason
#     sample_ids = math_ops.cast(
#             math_ops.argmax(outputs, axis=-1), dtypes.int32)
#     return sample_ids

#   def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
#     # Applies the fully connected pre-net to the decoder
#     # Also decides whether the decoder is finished
#     next_time = time + 1
#     finished = (next_time >= self._sequence_length)
#     all_finished = math_ops.reduce_all(finished)
#     def get_next_input(out):
#       # pre_net_result = self._pre_net_layer_2(self._pre_net_layer_1(outputs))
#       # next_input = tf.concat([pre_net_result, outputs], axis=-1)
#       next_input = self._pre_net_layer_2(self._pre_net_layer_1(outputs))
#       return next_input
#     # next_input =  nest.map_structure(read_from_ta, self._input_tas)
#     next_inputs = control_flow_ops.cond(
#         all_finished, 
#         lambda: self._zero_inputs,
#         lambda: get_next_input(outputs))

#     # print(next_input.shape)
#     # print(next_inputs.shape)
#     # input()

#     return (finished, next_inputs, state)

class TacotronTrainingHelper(Helper):
  """Base abstract class that allows the user to customize sampling."""

  def __init__(self, inputs, sequence_length, enable_prenet, 
              prenet_units=None, prenet_layers=None, sampling_prob=0., anneal_sampling_prob = False,
              time_major=False, sample_ids_shape=None, sample_ids_dtype=None, name=None,
              context=None):
    """Initializer.
    Args:
      initialize_fn: callable that returns `(finished, next_inputs)`
        for the first iteration.
      sample_fn: callable that takes `(time, outputs, state)`
        and emits tensor `sample_ids`.
      next_inputs_fn: callable that takes `(time, outputs, state, sample_ids)`
        and emits `(finished, next_inputs, next_state)`.
      sample_ids_shape: Either a list of integers, or a 1-D Tensor of type
        `int32`, the shape of each value in the `sample_ids` batch. Defaults to
        a scalar.
      sample_ids_dtype: The dtype of the `sample_ids` tensor. Defaults to int32.
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
    # if anneal_sampling_prob:
      ## Currently hard-coded
      # curr_epoch = tf.div(tf.cast(tf.train.get_or_create_global_step(),tf.float32), tf.constant(128./32.))
      # curr_step = tf.floor(tf.div(curr_epoch,tf.constant(100./20.)))
      # self.sampling_prob = tf.div(curr_step,tf.constant(20.))
    # else:
    self.sampling_prob = sampling_prob
    # self.context = context

    ## Create finished projection layer
    # self._finished_proj_layer = tf.layers.Dense(
    #   units=1,
    #   activation=tf.nn.sigmoid,
    #   use_bias=False,
    # )
    ## Create dense pre_net
    self.prenet_layers=[]
    if enable_prenet:
      for idx in range(prenet_layers):
        self.prenet_layers.append(tf.layers.Dense(
          name="prenet_{}".format(idx + 1),
          units=prenet_units,
          activation=tf.nn.relu,
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
    # next_inputs = self._input_tas.read(0)
    return (finished, self._zero_inputs )

  def sample(self, time, outputs, state, name=None):
    # Fully deterministic, output should already be projected
    del time, state
    # Required to appease tensorflow for some reason
    sample_ids = math_ops.cast(
            math_ops.argmax(outputs, axis=-1), dtypes.int32)
    return sample_ids

  def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
    # Applies the fully connected pre-net to the decoder
    # Also decides whether the decoder is finished
    next_time = time + 1
    finished = (next_time >= self._sequence_length)
    all_finished = math_ops.reduce_all(finished)
    def get_next_input(inp, out):
      next_input = inp.read(time)
      # pre_net_result = self._pre_net_layer_2(self._pre_net_layer_1(outputs))
      # next_input = tf.concat([pre_net_result, inp], axis=-1)
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
      # next_input = tf.concat([next_input, self.context],axis=-1)
      return next_input

    # next_input =  nest.map_structure(read_from_ta, self._input_tas)
    next_inputs = control_flow_ops.cond(
        all_finished, 
        lambda: self._zero_inputs,
        lambda: get_next_input(self._input_tas, outputs))

    return (finished, next_inputs, state)


class TacotronHelper(Helper):
  """Base abstract class that allows the user to customize sampling."""

  def __init__(self, inputs, sequence_length, enable_prenet, 
              prenet_units=None, prenet_layers=None,
              time_major=False, sample_ids_shape=None, sample_ids_dtype=None, name=None,
              context=None):
    """Initializer.
    Args:
      initialize_fn: callable that returns `(finished, next_inputs)`
        for the first iteration.
      sample_fn: callable that takes `(time, outputs, state)`
        and emits tensor `sample_ids`.
      next_inputs_fn: callable that takes `(time, outputs, state, sample_ids)`
        and emits `(finished, next_inputs, next_state)`.
      sample_ids_shape: Either a list of integers, or a 1-D Tensor of type
        `int32`, the shape of each value in the `sample_ids` batch. Defaults to
        a scalar.
      sample_ids_dtype: The dtype of the `sample_ids` tensor. Defaults to int32.
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
    # self.context = context

    self.prenet_layers=[]
    if enable_prenet:
      for idx in range(prenet_layers):
        self.prenet_layers.append(tf.layers.Dense(
          name="prenet_{}".format(idx + 1),
          units=prenet_units,
          activation=tf.nn.relu,
          use_bias=False,
        ))

      pre_net_output = tf.zeros([self._batch_size, prenet_units])
      self._zero_inputs = pre_net_output
    else:
      self._zero_inputs = nest.map_structure(
        lambda inp: array_ops.zeros_like(inp[0, :]), inputs)
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
    # next_inputs = self._input_tas.read(0)
    return (finished, self._zero_inputs)

  def sample(self, time, outputs, state, name=None):
    # Fully deterministic, output should already be projected
    del time, state
    # Required to appease tensorflow for some reason
    sample_ids = math_ops.cast(
            math_ops.argmax(outputs, axis=-1), dtypes.int32)
    return sample_ids

  def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
    # Applies the fully connected pre-net to the decoder
    # Also decides whether the decoder is finished
    next_time = time + 1
    finished = (next_time >= self._sequence_length)
    all_finished = math_ops.reduce_all(finished)

    def get_next_input(out):
      # pre_net_result = self._pre_net_layer_2(self._pre_net_layer_1(outputs))
      # next_input = tf.concat([pre_net_result, outputs], axis=-1)
      for layer in self.prenet_layers:
        out = tf.layers.dropout(layer(out), rate=0.5, training=True)
        # outputs = layer(outputs)
      # outputs = tf.concat([outputs, self.context],axis=-1)
      return out
    # next_input =  nest.map_structure(read_from_ta, self._input_tas)
    next_inputs = control_flow_ops.cond(
        all_finished, 
        lambda: self._zero_inputs,
        lambda: get_next_input(outputs))


    # print(next_input.shape)
    # print(next_inputs.shape)
    # input()

    return (finished, next_inputs, state)

class TrainingHelper(Helper):
  """A helper for use during training.  Only reads inputs.
  Returned sample_ids are the argmax of the RNN output logits.
  """

  def __init__(self, inputs, sequence_length, time_major=False, name=None):
    """Initializer.
    Args:
      inputs: A (structure of) input tensors.
      sequence_length: An int32 vector tensor.
      time_major: Python bool.  Whether the tensors in `inputs` are time major.
        If `False` (default), they are assumed to be batch major.
      name: Name scope for any created operations.
    Raises:
      ValueError: if `sequence_length` is not a 1D tensor.
    """
    with ops.name_scope(name, "TrainingHelper", [inputs, sequence_length]):
      inputs = ops.convert_to_tensor(inputs, name="inputs")
      self._inputs = inputs
      if not time_major:
        inputs = nest.map_structure(_transpose_batch_time, inputs)

      self._input_tas = nest.map_structure(_unstack_ta, inputs)
      self._sequence_length = ops.convert_to_tensor(
          sequence_length, name="sequence_length")
      if self._sequence_length.get_shape().ndims != 1:
        raise ValueError(
            "Expected sequence_length to be a vector, but received shape: %s" %
            self._sequence_length.get_shape())

      self._zero_inputs = nest.map_structure(
          lambda inp: array_ops.zeros_like(inp[0, :]), inputs)

      self._batch_size = array_ops.size(sequence_length)

  @property
  def inputs(self):
    return self._inputs

  @property
  def sequence_length(self):
    return self._sequence_length

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return tensor_shape.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return dtypes.int32

  def initialize(self, name=None):
    with ops.name_scope(name, "TrainingHelperInitialize"):
      finished = math_ops.equal(0, self._sequence_length)
      # all_finished = math_ops.reduce_all(finished)
      # next_inputs = control_flow_ops.cond(
      #     all_finished, lambda: self._zero_inputs,
      #     lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
      return (finished, self._zero_inputs)

  def sample(self, time, outputs, name=None, **unused_kwargs):
    with ops.name_scope(name, "TrainingHelperSample", [time, outputs]):
      sample_ids = math_ops.cast(
          math_ops.argmax(outputs, axis=-1), dtypes.int32)
      return sample_ids

  def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
    """next_inputs_fn for TrainingHelper."""
    with ops.name_scope(name, "TrainingHelperNextInputs",
                        [time, outputs, state]):
      next_time = time + 1
      finished = (next_time >= self._sequence_length)
      all_finished = math_ops.reduce_all(finished)
      def read_from_ta(inp):
        return inp.read(time)
      next_inputs = control_flow_ops.cond(
          all_finished, lambda: self._zero_inputs,
          lambda: nest.map_structure(read_from_ta, self._input_tas))
      return (finished, next_inputs, state)

class ScheduledSamplingHelper(TacotronTrainingHelper):
  """A helper for use during training.  Only reads inputs.
  Returned sample_ids are the argmax of the RNN output logits.
  """

  def __init__(self, inputs, sequence_length, sampling_probability, seed, time_major=False, name=None):
    """Initializer.
    Args:
      inputs: A (structure of) input tensors.
      sequence_length: An int32 vector tensor.
      time_major: Python bool.  Whether the tensors in `inputs` are time major.
        If `False` (default), they are assumed to be batch major.
      name: Name scope for any created operations.
    Raises:
      ValueError: if `sequence_length` is not a 1D tensor.
    """
    self._sampling_probability = ops.convert_to_tensor(
        sampling_probability, name="sampling_probability")
    if self._sampling_probability.get_shape().ndims not in (0, 1):
      raise ValueError(
          "sampling_probability must be either a scalar or a vector. "
          "saw shape: %s" % (self._sampling_probability.get_shape()))
    super(ScheduledSamplingHelper, self).__init__(
          inputs=inputs,
          sequence_length=sequence_length,
          time_major=time_major,
          name=name)
    self.seed = seed

  @property
  def inputs(self):
    return self._inputs

  @property
  def sequence_length(self):
    return self._sequence_length

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return tensor_shape.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return dtypes.int32

  def initialize(self, name=None):
    with ops.name_scope(name, "TrainingHelperInitialize"):
      return super(ScheduledSamplingHelper, self).initialize(name=name)

  def sample(self, time, outputs, state, name=None):
    with ops.name_scope(name, "ScheduledEmbeddingTrainingHelperSample", [time, outputs]):
      sample_ids = math_ops.cast(
          math_ops.argmax(outputs, axis=-1), dtypes.int32)
      return sample_ids

  def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
    """next_inputs_fn for TrainingHelper."""
    with ops.name_scope(name, "ScheduledEmbeddingTrainingHelperNextInputs",
                        [time, outputs, state]):
      (finished, base_next_inputs, state) = (
          super(ScheduledSamplingHelper, self).next_inputs(
              time=time,
              outputs=outputs,
              state=state,
              name=name))

      def maybe_sample(outputs):
        """Perform scheduled sampling."""
        outputs = self._pre_net_layer_2(self._pre_net_layer_1(outputs))
        select_sampler = bernoulli.Bernoulli(
          probs=self._sampling_probability, dtype=dtypes.bool)
        select_sample = select_sampler.sample(
            sample_shape=self.batch_size, seed=self.seed)
        select_sample = tf.reshape(tf.tile(select_sample, [256]), [32, 256])
        sample_ids = array_ops.where(
            select_sample,
            outputs,
            gen_array_ops.fill([self.batch_size, 256], -1.))
        where_sampling = math_ops.cast(
            array_ops.where(sample_ids > -1), dtypes.int32)
        where_not_sampling = math_ops.cast(
            array_ops.where(sample_ids <= -1), dtypes.int32)
        sample_ids_sampling = array_ops.gather_nd(sample_ids, where_sampling)
        inputs_not_sampling = array_ops.gather_nd(
            base_next_inputs, where_not_sampling)
        sampled_next_inputs = sample_ids_sampling
        base_shape = array_ops.shape(base_next_inputs)
        return (array_ops.scatter_nd(indices=where_sampling,
                                     updates=sampled_next_inputs,
                                     shape=base_shape)
                + array_ops.scatter_nd(indices=where_not_sampling,
                                       updates=inputs_not_sampling,
                                       shape=base_shape))

      all_finished = math_ops.reduce_all(finished)
      next_inputs = control_flow_ops.cond(
          all_finished, lambda: base_next_inputs, lambda: maybe_sample(outputs))
    return (finished, next_inputs, state)

class InferenceHelper(Helper):
  """A helper for use during training.  Only reads inputs.
  Returned sample_ids are the argmax of the RNN output logits.
  """

  def __init__(self, inputs, sequence_length, time_major=False, name=None):
    """Initializer.
    Args:
      inputs: A (structure of) input tensors.
      sequence_length: An int32 vector tensor.
      time_major: Python bool.  Whether the tensors in `inputs` are time major.
        If `False` (default), they are assumed to be batch major.
      name: Name scope for any created operations.
    Raises:
      ValueError: if `sequence_length` is not a 1D tensor.
    """
    with ops.name_scope(name, "InferenceHelper", [inputs, sequence_length]):
      inputs = ops.convert_to_tensor(inputs, name="inputs")
      self._inputs = inputs
      if not time_major:
        inputs = nest.map_structure(_transpose_batch_time, inputs)

      self._input_tas = nest.map_structure(_unstack_ta, inputs)
      self._sequence_length = ops.convert_to_tensor(
          sequence_length, name="sequence_length")
      if self._sequence_length.get_shape().ndims != 1:
        raise ValueError(
            "Expected sequence_length to be a vector, but received shape: %s" %
            self._sequence_length.get_shape())

      self._zero_inputs = nest.map_structure(
          lambda inp: array_ops.zeros_like(inp[0, :]), inputs)

      self._batch_size = array_ops.size(sequence_length)

  @property
  def inputs(self):
    return self._inputs

  @property
  def sequence_length(self):
    return self._sequence_length

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return tensor_shape.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return dtypes.int32

  def initialize(self, name=None):
    with ops.name_scope(name, "InferenceHelperInitialize"):
      finished = math_ops.equal(0, self._sequence_length)
      # all_finished = math_ops.reduce_all(finished)
      # next_inputs = control_flow_ops.cond(
      #     all_finished, lambda: self._zero_inputs,
      #     lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
      return (finished, self._zero_inputs)

  def sample(self, time, outputs, name=None, **unused_kwargs):
    with ops.name_scope(name, "InferenceHelperSample", [time, outputs]):
      sample_ids = math_ops.cast(
          math_ops.argmax(outputs, axis=-1), dtypes.int32)
      return sample_ids

  def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
    """next_inputs_fn for TrainingHelper."""
    with ops.name_scope(name, "InferenceHelperNextInputs",
                        [time, outputs, state]):
      next_time = time + 1
      finished = (time >= self._sequence_length)
      all_finished = math_ops.reduce_all(finished)
      next_inputs = control_flow_ops.cond(
          all_finished, lambda: self._zero_inputs,
          lambda: outputs)
      return (finished, next_inputs, state)