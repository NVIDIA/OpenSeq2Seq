# Copyright (c) 2017 NVIDIA Corporation
"""
This module implements attention mechanisms described in
"Attention is All You Need" https://arxiv.org/abs/1706.03762
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
from .common import inf

def scaled_dot_attention_fn(Q,
                            K,
                            V,
                            sqrt_normalize=True,
                            bias=None):
  """
  Computes scaled dot attention (formula 1 from Section 3.2.1 in the
  paper above). Per batch per head
  Contains no trainable parameters
  :param Q: Queries tensor [batch, num_heads, Q_length, dk]
  :param K: Keys tensor [batch, num_heads, K_length, dk]
  :param V: Values tensor [batch, num_heads, V_length, dv]
  :param sqrt_normalize: (default: True) whether to normalize by sqrt(dk)
  :param bias: (default: None) masking bias
  :return: scaled dot attention tensor of shape [batch, num_heads, length_q, dv]
  """
  dk = Q.shape[-1].value or tf.shape(Q)[-1]  # last dimension
  assert(dk == K.shape[-1].value or tf.shape(K)[-1])
  with tf.name_scope("ScaledDotAttention"):
    #logits = tf.cast(tf.matmul(Q, K, transpose_b=True), dtype=tf.float32)
    logits = tf.matmul(Q, K, transpose_b=True)
    if sqrt_normalize:
      softmax_input = tf.scalar_mul(
        scalar=tf.sqrt(tf.cast(dk, dtype=logits.dtype)),
        x=logits)
    else:
      softmax_input = logits

    if bias is not None:
      #softmax_input += tf.cast(bias, dtype=logits.dtype)
      softmax_input += bias
    else:
      softmax_input = logits

    return tf.matmul(tf.nn.softmax(softmax_input), V)

def get_future_masking_bias(Q, K):
  """
  Performs future masking for decoder, by setting everything >
  current position to -INF (-1e9).
  It asserts that Q and K are of the same shape
  :param Q: Queries tensor [batch, num_heads, Q_length, dk]
  :param K: Keys tensor [batch, num_heads, K_length, dk]
  :return: tensor with same dtype as Q and
  of shape [batch, num_heads, Q_length, K_length],
  where for all batch_ind and head_ind:
      [batch_ind, head_ind, :, :] is an upper diagonal (without diagonal) and
      all non zero entries are -INF
  """
  tf.assert_equal(tf.shape(Q), tf.shape(K))
  shape = [tf.shape(Q)[0], tf.shape(Q)[1], tf.shape(Q)[2], tf.shape(K)[2]]
  return tf.cast((tf.matrix_band_part(tf.ones(shape=shape), 0, -1) -
          tf.matrix_band_part(tf.ones(shape=shape), 0, 0))*inf, dtype=Q.dtype)

def multi_head_attention_fn(Q,
                            K,
                            V,
                            d_model,
                            dk=None,
                            dv=None,
                            h=2,
                            mask_future=False,
                            additional_bias=None,
                            initializer=None):
  """
  Computes multi-head attention (sess Section 3.2. in the paper above)
  :param Q: Queries tensor [batch, Q_length, orig_dq]
  :param K: Keys tensor [batch, K_length, orig_dk]
  :param V: Values tensor [batch, V_length, orig_dv]
  :param d_model: model dimensionality
  :param dk: (default: d_model/h) Q and K will be projected to dk
  :param dv: (default: d_model/h) V will be projected to dv
  :param h: (default: 2) number of heads in attention
  :param mask_future: (default: False) whether to mask future steps
  :param additional_bias: (default: None) additional bias, such as pad masking bias
  should be a tensor of shape [batch, heads, Q_length, K_length]
  :param initializer: (default: None) initializer for projection
  :return:
  """
  if dk is None:
    dk = int(d_model/h)
  if dv is None:
    dv = int(d_model/h)
  with tf.variable_scope("MultiHeadAttention"):
    Q_multi_head = tf.stack(
      tf.split(tf.layers.dense(inputs=Q,
                               units=dk*h,
                               use_bias=False,
                               name="Q_proj",
                               kernel_initializer=initializer),
               num_or_size_splits=h,
               axis=-1),
      axis=1)
    K_multi_head = tf.stack(
      tf.split(tf.layers.dense(inputs=K,
                               units=dk*h,
                               use_bias=False,
                               name="K_proj",
                               kernel_initializer=initializer),
               num_or_size_splits=h,
               axis=-1),
      axis=1)
    V_multi_head = tf.stack(
      tf.split(tf.layers.dense(inputs=V,
                               units=dv*h,
                               use_bias=False,
                               name="V_proj",
                               kernel_initializer=initializer),
               num_or_size_splits=h,
               axis=-1),
      axis=1)
    
    # now, Q, K and V are 4-dimensional [batch, num_heads, length, dim]
    if mask_future is False:
      bias = additional_bias # can be None
    else: # mask future
      # future_masking_bias = get_future_masking_bias(Q_multi_head, K_multi_head)
      future_masking_bias = get_future_masking_bias(K_multi_head, V_multi_head)
      if additional_bias is not None:
        bias = additional_bias + future_masking_bias
      else:
        bias = future_masking_bias
    heads = scaled_dot_attention_fn(Q=Q_multi_head,
                                    K=K_multi_head,
                                    V=V_multi_head,
                                    bias=bias)

    # heads are of shape [batch, num_heads, length_q, dv]
    sq_values = [tf.squeeze(t) for t in tf.split(heads,
                                                 num_or_size_splits=h,
                                                 axis=1)]
    result = tf.layers.dense(inputs=tf.reshape(
      tf.concat(values=sq_values,
                axis=-1),
      shape=[tf.shape(Q)[0], -1, h*dv]),
      units=d_model,
      use_bias=False,
      name="W_O")
    return result




