from __future__ import absolute_import, division, print_function
import tensorflow as tf
from open_seq2seq.parts.t2t_timing_signal import add_timing_signal
from open_seq2seq.data.text2text import SpecialTextTokens

inf = -1e4
#inf = 0.0

def normalize(input):
  return tf.contrib.layers.layer_norm(input,
                                      begin_norm_axis=-1,
                                      begin_params_axis=-1)


def dropout_normalize_add_btd(output, input=None, drop_prob=0.0):
  """
  Performs dropout on output, normalizes and adds input
  :param output:
  :param input:
  :param drop_prob:
  :return:
  """
  if drop_prob != 0.0:
    output = tf.nn.dropout(x=output, keep_prob=1-drop_prob,
                           noise_shape=[tf.shape(output)[0], 1, tf.shape(output)[2]])
  # residual connection
  if input is not None:
    output += input

  return normalize(output)


def get_pad_masking_bias(x, y, PAD_ID, heads, dtype=tf.float32):
  """
  :param src: a tensor of shape [batch_size, x_len]
  :param src: a tensor of shape [batch_size, y_len]
  :return: a tensor of shape [batch, heads, x_len, y_len]
  """
  maskQ = tf.to_float(tf.not_equal(x, PAD_ID))
  maskK = tf.to_float(tf.not_equal(y, PAD_ID))
  #maskQ = tf.not_equal(x, PAD_ID)
  #maskK = tf.not_equal(y, PAD_ID)
  attention_bias = tf.matmul(tf.expand_dims(maskQ, -1), tf.expand_dims(maskK, 1))
  attention_bias = tf.expand_dims(attention_bias, 1) # add dimension for heads
  attention_bias = tf.tile(attention_bias, multiples=[1, heads, 1, 1])
  attention_bias = tf.to_float(tf.equal(attention_bias, 0))
  #attention_bias = tf.cast(tf.equal(attention_bias, 0))
  #attention_bias = tf.Print(attention_bias,
  #                          [tf.reduce_max(tf.abs(attention_bias))], message="Before")
  attention_bias = tf.scalar_mul(scalar=inf, x=attention_bias)
  #attention_bias = tf.Print(attention_bias, [tf.reduce_max(tf.abs(attention_bias))], message="After")
  return tf.cast(attention_bias, dtype=dtype)


def ffn_and_layer_norm(inpt,
                       inner_dim,
                       resulting_dim,
                       drop_prob=0.0,
                       inner_activation=tf.nn.relu):
  """
  Position-wise fully connected feed-forward network with layer norm
  and residual connection
  :param inpt: input tensor
  :param inner_dim: bottleneck dimension
  :param resulting_dim: output dimensionality
  :param drop_prob: dropout probability of drop
  :param inner_activation: inner activation function
  :return:
  """
  with tf.variable_scope("FFN_and_layer_norm"):
    inner_act = tf.layers.dense(inputs=inpt,
                                units=inner_dim,
                                activation=inner_activation,
                                name="first_dense")
    ffn_out = tf.layers.dense(inputs=inner_act,
                              units=resulting_dim,
                              activation=None,
                              name="second_dense")

    res = dropout_normalize_add_btd(output=ffn_out, input=inpt,
                                    drop_prob=drop_prob)
    return res


def embed_and_maybe_add_position_signal(inpt,
                                        emb_W,
                                        num_timescales,                                        
                                        heads,
                                        d_model=None):
  with tf.name_scope("embed_and_maybe_add_position_signal"):
    with tf.name_scope("embed"):
      embedded_inputs = tf.nn.embedding_lookup(emb_W,
                                             inpt)

    with tf.name_scope("pad_masking"):
      bias = get_pad_masking_bias(inpt, inpt, PAD_ID=SpecialTextTokens.PAD_ID.value,
                                  heads=heads, dtype=emb_W.dtype)

    with tf.name_scope("timing_signal"):
      if num_timescales != 0:
        embedded_inputs_with_pos = add_timing_signal(embedded_inputs,
                                                     num_timescales=num_timescales)
      else:
        return embedded_inputs, bias
      return embedded_inputs_with_pos, bias
