# pylint: skip-file
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
"""Attention for convolutional decoder
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from six.moves import range

import collections
import functools
import math

import numpy as np

from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.layers.convolutional import Conv1D
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

__all__ = [
    "AttentionMechanism", "BahdanauAttention"
]

class AttentionMechanism(object):

  @property
  def alignments_size(self):
    raise NotImplementedError

  @property
  def state_size(self):
    raise NotImplementedError


def _prepare_memory(memory, memory_sequence_length, check_inner_dims_defined):
  """Convert to tensor and possibly mask `memory`.

  Args:
    memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
    memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
    check_inner_dims_defined: Python boolean.  If `True`, the `memory`
      argument's shape is checked to ensure all but the two outermost
      dimensions are fully defined.

  Returns:
    A (possibly masked), checked, new `memory`.

  Raises:
    ValueError: If `check_inner_dims_defined` is `True` and not
      `memory.shape[2:].is_fully_defined()`.
  """
  memory = nest.map_structure(
      lambda m: ops.convert_to_tensor(m, name="memory"), memory
  )
  if memory_sequence_length is not None:
    memory_sequence_length = ops.convert_to_tensor(
        memory_sequence_length, name="memory_sequence_length"
    )
  if check_inner_dims_defined:

    def _check_dims(m):
      if not m.get_shape()[2:].is_fully_defined():
        raise ValueError(
            "Expected memory %s to have fully defined inner dims, "
            "but saw shape: %s" % (m.name, m.get_shape())
        )

    nest.map_structure(_check_dims, memory)
  if memory_sequence_length is None:
    seq_len_mask = None
  else:
    seq_len_mask = array_ops.sequence_mask(
        memory_sequence_length,
        maxlen=array_ops.shape(nest.flatten(memory)[0])[1],
        dtype=nest.flatten(memory)[0].dtype
    )
    seq_len_batch_size = (
        memory_sequence_length.shape[0].value or
        array_ops.shape(memory_sequence_length)[0]
    )

  def _maybe_mask(m, seq_len_mask):
    rank = m.get_shape().ndims
    rank = rank if rank is not None else array_ops.rank(m)
    extra_ones = array_ops.ones(rank - 2, dtype=dtypes.int32)
    m_batch_size = m.shape[0].value or array_ops.shape(m)[0]
    if memory_sequence_length is not None:
      message = (
          "memory_sequence_length and memory tensor batch sizes do not "
          "match."
      )
      with ops.control_dependencies(
          [
              check_ops.assert_equal(
                  seq_len_batch_size, m_batch_size, message=message
              )
          ]
      ):
        seq_len_mask = array_ops.reshape(
            seq_len_mask,
            array_ops.concat((array_ops.shape(seq_len_mask), extra_ones), 0)
        )
        return m * seq_len_mask
    else:
      return m

  return nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask), memory)


def _maybe_mask_score(score, memory_sequence_length, score_mask_value):
  if memory_sequence_length is None:
    return score
  message = ("All values in memory_sequence_length must greater than zero.")
  with ops.control_dependencies(
      [check_ops.assert_positive(memory_sequence_length, message=message)]
  ):
    score_mask = array_ops.sequence_mask(
        memory_sequence_length, maxlen=array_ops.shape(score)[-1]
    )
    if score.get_shape().ndims == 3:
      max_len = score.shape[1].value or array_ops.shape(score)[1]
      score_mask = array_ops.expand_dims(score_mask, 1)
      score_mask = array_ops.tile(score_mask, [1, max_len, 1])
    score_mask_values = score_mask_value * array_ops.ones_like(score)
    return array_ops.where(score_mask, score, score_mask_values)


class _BaseAttentionMechanism(AttentionMechanism):
  """A base AttentionMechanism class providing common functionality.

  Common functionality includes:
    1. Storing the query and memory layers.
    2. Preprocessing and storing the memory.
  """

  def __init__(
      self,
      query_layer,
      memory,
      probability_fn,
      memory_sequence_length=None,
      memory_layer=None,
      check_inner_dims_defined=True,
      score_mask_value=None,
      name=None
  ):
    """Construct base AttentionMechanism class.

    Args:
      query_layer: Callable.  Instance of `tf.layers.Layer`.  The layer's depth
        must match the depth of `memory_layer`.  If `query_layer` is not
        provided, the shape of `query` must match that of `memory_layer`.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      probability_fn: A `callable`.  Converts the score and previous alignments
        to probabilities. Its signature should be:
        `probabilities = probability_fn(score, state)`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      memory_layer: Instance of `tf.layers.Layer` (may be None).  The layer's
        depth must match the depth of `query_layer`.
        If `memory_layer` is not provided, the shape of `memory` must match
        that of `query_layer`.
      check_inner_dims_defined: Python boolean.  If `True`, the `memory`
        argument's shape is checked to ensure all but the two outermost
        dimensions are fully defined.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      name: Name to use when creating ops.
    """
    if (
        query_layer is not None and
        not isinstance(query_layer, layers_base.Layer)
    ):
      raise TypeError(
          "query_layer is not a Layer: %s" % type(query_layer).__name__
      )
    if (
        memory_layer is not None and
        not isinstance(memory_layer, layers_base.Layer)
    ):
      raise TypeError(
          "memory_layer is not a Layer: %s" % type(memory_layer).__name__
      )
    self._query_layer = query_layer
    self._memory_layer = memory_layer
    self.dtype = memory_layer.dtype
    if not callable(probability_fn):
      raise TypeError(
          "probability_fn must be callable, saw type: %s" %
          type(probability_fn).__name__
      )
    if score_mask_value is None:
      score_mask_value = dtypes.as_dtype(self._memory_layer.dtype
                                        ).as_numpy_dtype(-np.inf)
    self._probability_fn = lambda score, prev: (  # pylint:disable=g-long-lambda
        probability_fn(
            _maybe_mask_score(score, memory_sequence_length, score_mask_value),
            prev))
    with ops.name_scope(
        name, "BaseAttentionMechanismInit", nest.flatten(memory)
    ):
      self._values = _prepare_memory(
          memory,
          memory_sequence_length,
          check_inner_dims_defined=check_inner_dims_defined
      )
      self._keys = (
          self.memory_layer(self._values) if self.memory_layer  # pylint: disable=not-callable
          else self._values
      )
      self._batch_size = (
          self._keys.shape[0].value or array_ops.shape(self._keys)[0]
      )
      self._alignments_size = (
          self._keys.shape[1].value or array_ops.shape(self._keys)[1]
      )

  @property
  def memory_layer(self):
    return self._memory_layer

  @property
  def query_layer(self):
    return self._query_layer

  @property
  def values(self):
    return self._values

  @property
  def keys(self):
    return self._keys

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def alignments_size(self):
    return self._alignments_size

  @property
  def state_size(self):
    return self._alignments_size

  def initial_alignments(self, batch_size, dtype):
    """Creates the initial alignment values for the `AttentionWrapper` class.

    This is important for AttentionMechanisms that use the previous alignment
    to calculate the alignment at the next time step (e.g. monotonic attention).

    The default behavior is to return a tensor of all zeros.

    Args:
      batch_size: `int32` scalar, the batch_size.
      dtype: The `dtype`.

    Returns:
      A `dtype` tensor shaped `[batch_size, alignments_size]`
      (`alignments_size` is the values' `max_time`).
    """
    max_time = self._alignments_size
    return _zero_state_tensors(max_time, batch_size, dtype)

  def initial_state(self, batch_size, dtype):
    """Creates the initial state values for the `AttentionWrapper` class.

    This is important for AttentionMechanisms that use the previous alignment
    to calculate the alignment at the next time step (e.g. monotonic attention).

    The default behavior is to return the same output as initial_alignments.

    Args:
      batch_size: `int32` scalar, the batch_size.
      dtype: The `dtype`.

    Returns:
      A structure of all-zero tensors with shapes as described by `state_size`.
    """
    return self.initial_alignments(batch_size, dtype)


def _bahdanau_score(processed_query, keys, normalize):
  """Implements Bahdanau-style (additive) scoring function.

  This attention has two forms.  The first is Bhandanau attention,
  as described in:

  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473

  The second is the normalized form.  This form is inspired by the
  weight normalization article:

  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868

  To enable the second form, set `normalize=True`.

  Args:
    processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys. Change
    keys: Processed memory, shape `[batch_size, max_time, num_units]`.
    normalize: Whether to normalize the score function.

  Returns:
    A `[batch_size, max_time]` tensor of unnormalized score values.
  """
  dtype = processed_query.dtype
  # Get the number of hidden units from the trailing dimension of keys
  num_units = keys.shape[2].value or array_ops.shape(keys)[2]
  # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
  query_rank = processed_query.get_shape().ndims

  if query_rank == 2:
    processed_query = array_ops.expand_dims(processed_query, 1)
  elif query_rank == 3:
    processed_query = array_ops.expand_dims(processed_query, 2)
    keys = array_ops.expand_dims(keys, 1)
  else:
    raise ValueError("The rank of the query should be either 2 or 3 but recieved {}".format(query_rank))

  v = variable_scope.get_variable("attention_v", [num_units], dtype=dtype)
  if normalize:
    # Scalar used in weight normalization
    g = variable_scope.get_variable(
        "attention_g",
        dtype=dtype,
        shape=[1],
        #initializer=math.sqrt((1. / num_units)))
        initializer=init_ops.constant_initializer(
            math.sqrt(1. / num_units), dtype=dtype
        )
    )
    # Bias added prior to the nonlinearity
    b = variable_scope.get_variable(
        "attention_b", [num_units],
        dtype=dtype,
        initializer=init_ops.zeros_initializer()
    )
    # normed_v = g * v / ||v||
    normed_v = g * v * math_ops.rsqrt(math_ops.reduce_sum(math_ops.square(v)))
    return math_ops.reduce_sum(
        normed_v * math_ops.tanh(keys + processed_query + b), [-1]
    )
  else:
    return math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query), [-1])


class BahdanauAttention(_BaseAttentionMechanism):
  """Implements Bahdanau-style (additive) attention.

  This attention has two forms.  The first is Bahdanau attention,
  as described in:

  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473

  The second is the normalized form.  This form is inspired by the
  weight normalization article:

  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868

  To enable the second form, construct the object with parameter
  `normalize=True`.
  """

  def __init__(
      self,
      num_units,
      memory,
      memory_sequence_length=None,
      normalize=False,
      probability_fn=None,
      score_mask_value=None,
      dtype=None,
      name="BahdanauAttention"
  ):
    """Construct the Attention mechanism.

    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      normalize: Python boolean.  Whether to normalize the energy term.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is @{tf.nn.softmax}. Other options include
        @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      dtype: The data type for the query and memory layers of the attention
        mechanism.
      name: Name to use when creating ops.
    """
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    if dtype is None:
      dtype = dtypes.float32
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(BahdanauAttention, self).__init__(
        query_layer=layers_core.Dense(
            num_units, name="query_layer", use_bias=False, dtype=dtype
        ),
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False, dtype=dtype
        ),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name
    )
    self._num_units = num_units
    self._normalize = normalize
    self._name = name

  def __call__(self, query, state):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      state: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).

    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """
    with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
      if query.get_shape().ndims == 3:
        batch_size = query.shape[0].value or array_ops.shape(query)[0]
        max_len = query.shape[1].value or array_ops.shape(query)[1]
        hidden_dim = query.shape[2].value or array_ops.shape(query)[2]
        query = array_ops.reshape(query, [-1, hidden_dim])
        processed_query = self.query_layer(query) if self.query_layer else query
        processed_query = array_ops.reshape(processed_query, [batch_size, max_len, -1])
        print(processed_query)

      score = _bahdanau_score(processed_query, self._keys, self._normalize)
    alignments = self._probability_fn(score, state)
    next_state = alignments
    return alignments, next_state, self._values


def hardmax(logits, name=None):
  """Returns batched one-hot vectors.

  The depth index containing the `1` is that of the maximum logit value.

  Args:
    logits: A batch tensor of logit values.
    name: Name to use when creating ops.
  Returns:
    A batched one-hot tensor.
  """
  with ops.name_scope(name, "Hardmax", [logits]):
    logits = ops.convert_to_tensor(logits, name="logits")
    if logits.get_shape()[-1].value is not None:
      depth = logits.get_shape()[-1].value
    else:
      depth = array_ops.shape(logits)[-1]
    return array_ops.one_hot(
        math_ops.argmax(logits, -1), depth, dtype=logits.dtype
    )


def _compute_attention(
    attention_mechanism, cell_output, attention_state, attention_layer
):
  """Computes the attention and alignments for a given attention_mechanism."""
  alignments, next_attention_state, values = attention_mechanism(
      cell_output, state=attention_state
  )

  if alignments.get_shape().ndims == 3:
    max_len = alignments.shape[1].value or array_ops.shape(alignments)[1]
    values = array_ops.expand_dims(values, 1)
    values = array_ops.tile(values, [1, max_len, 1, 1])

  # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
  expanded_alignments = array_ops.expand_dims(alignments, -2)
  # Context is the inner product of alignments and values along the
  # memory time dimension.
  # alignments shape is
  #   [batch_size, 1, memory_time]
  # attention_mechanism.values shape is
  #   [batch_size, memory_time, memory_size]
  # the batched matmul is over memory_time, so the output shape is
  #   [batch_size, 1, memory_size].
  # we then squeeze out the singleton dim.
  print(expanded_alignments)
  print(values)
  context = math_ops.matmul(expanded_alignments, values)
  print(context)
  context = array_ops.squeeze(context, [-2])

  if attention_layer is not None:
    attention = attention_layer(array_ops.concat([cell_output, context], -1))
  else:
    attention = context

  return attention, alignments, next_attention_state

