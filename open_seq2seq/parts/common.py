from __future__ import absolute_import, division, print_function
import tensorflow as tf
from open_seq2seq.parts.t2t_timing_signal import add_timing_signal
from open_seq2seq.data.text2text import SpecialTextTokens


def normalize(input):
  return tf.contrib.layers.layer_norm(input,
                                      begin_norm_axis=-1,
                                      begin_params_axis=-1)


def get_pad_masking_bias(x, y, PAD_ID, heads):
  """
  :param src: a tensor of shape [batch_size, x_len]
  :param src: a tensor of shape [batch_size, y_len]
  :return: a tensor of shape [batch, heads, x_len, y_len]
  """
  inf = -1e9
  maskQ = tf.to_float(tf.not_equal(x, PAD_ID))
  maskK = tf.to_float(tf.not_equal(y, PAD_ID))
  attention_bias = tf.matmul(tf.expand_dims(maskQ, -1), tf.expand_dims(maskK, 1))
  attention_bias = tf.expand_dims(attention_bias, 1) # add dimension for heads
  attention_bias = tf.tile(attention_bias, multiples=[1, heads, 1, 1])
  attention_bias = tf.to_float(tf.equal(attention_bias, 0))
  attention_bias = tf.scalar_mul(scalar=inf, x=attention_bias)
  return attention_bias


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
    ffn_out = tf.layers.dropout(inputs=ffn_out, rate=drop_prob,
                                noise_shape=[tf.shape(ffn_out)[0], 1, resulting_dim])
    res = normalize(ffn_out + inpt)
    return res


def embed_and_maybe_add_position_signal(inpt,
                                        emb_W,
                                        num_timescales,                                        
                                        heads,
                                        d_model=None):
  embedded_inputs = tf.nn.embedding_lookup(emb_W,
                                           inpt)

  bias = get_pad_masking_bias(inpt, inpt, PAD_ID=SpecialTextTokens.PAD_ID.value,
                              heads=heads)

  if num_timescales != 0:
    embedded_inputs_with_pos = add_timing_signal(embedded_inputs,
                                                 num_timescales=num_timescales)
  else:
    return embedded_inputs, bias
  return embedded_inputs_with_pos, bias