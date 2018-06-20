from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_NEG_INF = -1e9


def get_padding(x, padding_value=0):
  """Return float tensor representing the padding values in x.

  Args:
    x: int tensor with any shape
    padding_value: int value that

  Returns:
    flaot tensor with same shape as x containing boolean values.
      False -> non-padding, True -> padding
  """
  with tf.name_scope("padding"):
    return tf.equal(x, padding_value)


def get_padding_bias(x):
  """Calculate bias tensor from padding values in tensor.

  Bias tensor that is added to the pre-softmax logits,
  which has shape [batch_size, 1, length]. The tensor is zero at
  non-padding locations, and -1e9 (negative infinity) at padding locations.

  Args:
    x: int tensor with shape [batch_size, length]

  Returns:
    Attention bias tensor of shape [batch_size, 1, length].
  """
  with tf.name_scope("attention_bias"):
    padding = tf.cast(get_padding(x), dtype=tf.float32)
    attention_bias = padding * _NEG_INF
    attention_bias = tf.expand_dims(attention_bias, axis=1)
  return attention_bias
