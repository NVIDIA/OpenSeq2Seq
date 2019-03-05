# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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

# Added some functions from Tensor2Tensor library:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
#
# Copyright 2019 The Tensor2Tensor Authors.
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

# Modifications by OpenSeq2Seq team

"""Implementation of multiheaded attention and self-attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i, dim in enumerate(static):
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def reshape_by_blocks(x, x_shape, memory_block_size):
  """Reshapes input by splitting its length over blocks of memory_block_size.
  Args:
    x: a Tensor with shape [batch, heads, length, depth]
    x_shape: tf.TensorShape of x.
    memory_block_size: Integer which divides length.
  Returns:
    Tensor with shape
    [batch, heads, length // memory_block_size, memory_block_size, depth].
  """
  x = tf.reshape(x, [
      x_shape[0], x_shape[1], x_shape[2] // memory_block_size,
      memory_block_size, x_shape[3]
  ])
  return x


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          depth,
                          dropout_rate=0.0,
                          name=None):
  """Dot-product attention.
  Args:
    q: Tensor with shape [..., length_q, depth_k].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
      match with q.
    bias: bias Tensor (see attention_bias())
    dropout_rate: a float.
  Returns:
    Tensor with shape [..., length_q, depth_v].
  """
  with tf.variable_scope(
    name, default_name="dot_product_attention", values=[q, k, v]) as scope:
    q *= depth ** -0.5
    logits = tf.matmul(q, k, transpose_b=True)

    dtype = logits.dtype
    if dtype != tf.float32:
      # upcast softmax inputs
      logits = tf.cast(x=logits, dtype=tf.float32)
      if bias is not None:
        logits += bias
      weights = tf.nn.softmax(logits, name="attention_weights")
      # downcast softmax output
      weights = tf.cast(weights, dtype=dtype)
    else:
      if bias is not None:
        logits += bias
      weights = tf.nn.softmax(logits, name="attention_weights")

    if dropout_rate != 0.0:
      weights = tf.nn.dropout(weights, keep_prob=1 - dropout_rate)
    attention_output = tf.matmul(weights, v)
    return attention_output


def local_attention_1d(q, k, v, depth, block_length=128, filter_width=100,
                       dropout_rate=0., bias=None, name=None):
  """
  Compute local attention.

  Args:
    q: a Tensor with shape [batch_size, num_heads, length, dk]
    k: a Tensor with shape [batch_size, num_heads, length, dk]
    v: a Tensor with shape [batch_size, num_heads, length, dv]
    bias:
    block_length:
    filter_width:
    name:

  Returns:

  """
  with tf.variable_scope(name, default_name="local_self_attention_1d",
                         values=[q, k, v]):
    q.get_shape()[:-1].assert_is_compatible_with(k.get_shape()[:-1])
    q.get_shape()[:-1].assert_is_compatible_with(v.get_shape()[:-1])
    batch_size, num_heads, original_length, _ = shape_list(q)

    def pad_to_multiple(x, pad_length):
      x_length = shape_list(x)[2]
      return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

    def pad_l_and_r(x, pad_length):
      return tf.pad(x, [[0, 0], [0, 0], [pad_length, pad_length], [0, 0]])

    # Set up query blocks.
    # [batch, heads, blocks_q, block_length, depth_k]
    q = pad_to_multiple(q, block_length)
    q = reshape_by_blocks(q, shape_list(q), block_length)
    total_query_blocks = shape_list(q)[2]

    # Set up key and value blocks.
    # [batch, heads, blocks_k, block_length, depth_k]
    blocks_per_filter_width = filter_width // block_length
    remaining_items = filter_width % block_length
    k = pad_to_multiple(k, block_length)
    v = pad_to_multiple(v, block_length)
    k = pad_l_and_r(k, filter_width + block_length - remaining_items)
    v = pad_l_and_r(v, filter_width + block_length - remaining_items)
    k = reshape_by_blocks(k, shape_list(k), block_length)
    v = reshape_by_blocks(v, shape_list(v), block_length)

    total_kv_blocks = shape_list(k)[2]

    slices = []
    # prepare the left-most and right-most partial blocks if needed
    if remaining_items:
      first_partial_block_k = tf.slice(
        k, [0, 0, 0, block_length - remaining_items, 0],
        [-1, -1, total_query_blocks, -1, -1])
      first_partial_block_v = tf.slice(
        v, [0, 0, 0, block_length - remaining_items, 0],
        [-1, -1, total_query_blocks, -1, -1])
      last_partial_block_k = tf.slice(
        k, [0, 0, total_kv_blocks - total_query_blocks, 0, 0],
        [-1, -1, -1, remaining_items, -1])
      last_partial_block_v = tf.slice(
        v, [0, 0, total_kv_blocks - total_query_blocks, 0, 0],
        [-1, -1, -1, remaining_items, -1])
      slices.append((first_partial_block_k, first_partial_block_v))
      slices.append((last_partial_block_k, last_partial_block_v))

    # Prepare the rest of the blocks
    first_block_index = 1 if remaining_items else 0
    attention_blocks = 2 * blocks_per_filter_width + 1
    for i in range(first_block_index, attention_blocks + first_block_index):
      block_k = tf.slice(k, [0, 0, i, 0, 0],
                         [-1, -1, total_query_blocks, -1, -1])
      block_v = tf.slice(v, [0, 0, i, 0, 0],
                         [-1, -1, total_query_blocks, -1, -1])
      slices.append((block_k, block_v))
    # [batch, heads, blocks_q, block_length + 2 * filter_width, depth_k]
    k = tf.concat([s[0] for s in slices], axis=3)
    v = tf.concat([s[1] for s in slices], axis=3)

    if bias is not None:
      attention_bias = tf.expand_dims(bias, axis=-2)
    else:
      attention_bias = None

    depth_v = shape_list(v)[-1]
    output = dot_product_attention(
        q,
        k,
        v,
        attention_bias,
        depth,
        dropout_rate=dropout_rate,
        name="local_1d")
    output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])

    # Remove the padding if introduced.
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output.set_shape([None if isinstance(dim, tf.Tensor) else dim for dim in
                      (batch_size, num_heads, original_length, depth_v)])
    return output




class Attention(tf.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(
      self,
      hidden_size,
      num_heads,
      attention_dropout,
      train,
      mode="loung",
      regularizer=None,
      block_length=None,
      filter_width=None,
  ):
    if hidden_size % num_heads != 0:
      raise ValueError("Hidden size must be evenly divisible by the number of "
                       "heads.")

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout
    self.train = train
    self.mode = mode

    self.block_length = block_length
    self.filter_width = filter_width
    if self.block_length is not None:
      assert(self.filter_width is not None)

    # Layers for linearly projecting the queries, keys, and values.
    self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q",
                                         kernel_regularizer=regularizer)
    self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k",
                                         kernel_regularizer=regularizer)
    self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v",
                                         kernel_regularizer=regularizer)
    self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                         name="output_transform",
                                         kernel_regularizer=regularizer)

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.

    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.

    Args:
      x: A tensor with shape [batch_size, length, hidden_size]

    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      # Calculate depth of last dimension after it has been split.
      depth = (self.hidden_size // self.num_heads)

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3])

  def combine_heads(self, x):
    """Combine tensor that has been split.

    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
      return tf.reshape(x, [batch_size, length, self.hidden_size])

  def call(self, x, y, bias, cache=None):
    """Apply attention mechanism to x and y.

    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.

    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    q = self.q_dense_layer(x)
    k = self.k_dense_layer(y)
    v = self.v_dense_layer(y)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      k = tf.concat([cache["k"], k], axis=1)
      v = tf.concat([cache["v"], v], axis=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    # Split q, k, v into heads.
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    # To scale q to prevent the dot product between q and k from growing too large.
    depth = (self.hidden_size // self.num_heads)

    if self.block_length is not None:
      attention_output = local_attention_1d(q=q, k=k, v=v,
                                            block_length=self.block_length,
                                            filter_width=self.filter_width,
                                            depth=depth,
                                            dropout_rate=0.0 if not self.train else self.attention_dropout,
                                            bias=bias)
    else: # Regular attention
      if self.mode == "loung":
        attention_output = dot_product_attention(
          q=q, k=k, v=v, bias=bias, depth=depth,
          dropout_rate=0.0 if not self.train else self.attention_dropout)
        # # Scale q to prevent the dot product between q and k from growing too large.
        # depth = (self.hidden_size // self.num_heads)
        # q *= depth ** -0.5
        # logits = tf.matmul(q, k, transpose_b=True)
        # dtype = logits.dtype
        # if dtype != tf.float32:
        #   # upcast softmax inputs
        #   logits = tf.cast(x=logits, dtype=tf.float32)
        #   if bias is not None:
        #     logits += bias
        #   weights = tf.nn.softmax(logits, name="attention_weights")
        #   # downcast softmax output
        #   weights = tf.cast(weights, dtype=dtype)
        # else:
        #   if bias is not None:
        #     logits += bias
        #   weights = tf.nn.softmax(logits, name="attention_weights")
      elif self.mode == "bahdanau":
        att_v = tf.get_variable(
            "attention_v", [self.hidden_size // self.num_heads], dtype=q.dtype
        )

        # Compute the attention score
        if bias is not None:
          weights = tf.reduce_sum(
              tf.nn.tanh(att_v * tf.nn.tanh(k + q + bias)), 3
          )
        else:
          weights = tf.reduce_sum(
              tf.nn.tanh(att_v * tf.nn.tanh(k + q)), 3
          )
        weights = tf.nn.softmax(weights)
        weights = tf.expand_dims(weights, 2)
        if self.train:
          weights = tf.nn.dropout(weights, keep_prob=1 - self.attention_dropout)
        attention_output = tf.matmul(weights, v)
      else:
        raise ValueError(
            "Mode for multi-head attention must be either loung for dot-product",
            "attention, or bahdanau for content-based/additive/mlp-base attention"
        )

    # Recombine heads --> [batch_size, length, hidden_size]
    attention_output = self.combine_heads(attention_output)

    # Run the combined outputs through another linear projection layer.
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self, x, bias, cache=None):
    return super(SelfAttention, self).call(x, x, bias, cache)
