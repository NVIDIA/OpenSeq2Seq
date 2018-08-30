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
"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import utils as model_utils


class EmbeddingSharedWeights(tf.layers.Layer):
  """Calculates input embeddings and pre-softmax linear with shared weights."""

  def __init__(self, vocab_size, hidden_size, pad_vocab_to_eight=False, init_var=None,
               embed_scale=True, pad_sym=0, mask_paddings=True, regularizer=None):
    super(EmbeddingSharedWeights, self).__init__()
    self.hidden_size = hidden_size
    self.embed_scale = embed_scale
    self.pad_sym = pad_sym
    self.mask_paddings = mask_paddings
    self.regularizer = regularizer

    padf = lambda x: x if x % 8 == 0 else x + 8 - x % 8
    if pad_vocab_to_eight:
      self.vocab_size = padf(vocab_size)
    else:
      self.vocab_size = vocab_size

    if init_var is None:
      self.init_var = hidden_size ** -0.5
    else:
      self.init_var = init_var

  def build(self, _):
    with tf.variable_scope("embedding_and_softmax", reuse=tf.AUTO_REUSE):
      # Create and initialize weights. The random normal initializer was chosen
      # randomly, and works well.
      self.shared_weights = tf.get_variable("weights", [self.vocab_size, self.hidden_size],
                                            initializer=tf.random_normal_initializer(0., self.init_var), \
                                            regularizer=self.regularizer)

    self.built = True

  def call(self, x):
    """Get token embeddings of x.

    Args:
      x: An int64 tensor with shape [batch_size, length]
    Returns:
      embeddings: float32 tensor with shape [batch_size, length, embedding_size]
      padding: float32 tensor with shape [batch_size, length] indicating the
        locations of the padding tokens in x.
    """
    with tf.name_scope("embedding"):
      # fills out of bound values with padding symbol
      out_bound_mask = tf.to_int32(x > (self.vocab_size - 1))
      x *= 1 - out_bound_mask
      x += out_bound_mask * tf.to_int32(self.pad_sym)

      embeddings = tf.gather(self.shared_weights, x)
      if self.embed_scale:
        # Scale embedding by the sqrt of the hidden size
        embeddings *= self.hidden_size ** 0.5

      if self.mask_paddings:
        # Create binary array of size [batch_size, length]
        # where 1 = padding, 0 = not padding
        padding = model_utils.get_padding(x, padding_value=self.pad_sym)

        # Set all padding embedding values to 0
        #embeddings *= tf.expand_dims(1 - padding, -1)
        embeddings *= tf.cast(tf.expand_dims(1.0 - padding, -1), dtype=embeddings.dtype)
      return embeddings

  def linear(self, x):
    """Computes logits by running x through a linear layer.

    Args:
      x: A float32 tensor with shape [batch_size, length, hidden_size]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
    with tf.name_scope("presoftmax_linear"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      x = tf.reshape(x, [-1, self.hidden_size])
      logits = tf.matmul(x, self.shared_weights, transpose_b=True)

      return tf.reshape(logits, [batch_size, length, self.vocab_size])
