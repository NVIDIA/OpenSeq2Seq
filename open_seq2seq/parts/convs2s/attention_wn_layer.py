"""Implementation of the attention layer for convs2s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from open_seq2seq.parts.convs2s.ffn_wn_layer import FeedFowardNetworkNormalized

class AttentionLayerNormalized(tf.layers.Layer):
  """Attention layer for convs2s with weight normalization"""
  """ Inspired from https://github.com/tobyyouup/conv_seq2seq"""

  def __init__(self, in_dim, embed_size, layer_id, add_res):
    super(AttentionLayerNormalized, self).__init__()

    self.add_res = add_res
    with tf.variable_scope("attention_layer_" + str(layer_id)):

      # linear projection layer to project the attention input to target space
      self.tgt_embed_proj = FeedFowardNetworkNormalized(in_dim, embed_size, dropout=1.0,
                                                          var_scope_name="att_linear_mapping_tgt_embed")

      # linear projection layer to project back to the input space
      self.out_proj = FeedFowardNetworkNormalized(embed_size, in_dim, dropout=1.0,
                                                    var_scope_name="att_linear_mapping_out")

  def call(self, input, target_embed, encoder_output_a, encoder_output_c, input_attention_bias):
    h_proj = self.tgt_embed_proj(input)
    d_proj = (h_proj + target_embed) * tf.sqrt(0.5)
    att_score = tf.matmul(d_proj, encoder_output_a, transpose_b=True)

    # mask out the paddings
    if input_attention_bias is not None:
      att_score = att_score + input_attention_bias

    att_score = tf.nn.softmax(att_score)

    length = tf.cast(tf.shape(encoder_output_c), tf.float32)
    output = tf.matmul(att_score, encoder_output_c) * length[1] * tf.sqrt(1.0 / length[1])
    output = self.out_proj(output)

    if self.add_res:
      output = (output + input) * tf.sqrt(0.5)

    return output
