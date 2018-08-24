"""Implementation of the attention layer for convs2s.
Inspired from https://github.com/tobyyouup/conv_seq2seq"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import math
from open_seq2seq.parts.convs2s.ffn_wn_layer import FeedFowardNetworkNormalized


class AttentionLayerNormalized(tf.layers.Layer):
  """Attention layer for convs2s with weight normalization"""

  def __init__(self, in_dim, embed_size, layer_id, add_res, mode,
               scaling_factor=math.sqrt(0.5),
               normalization_type="weight_norm",
               regularizer=None,
               init_var=None,
               ):
    """initializes the attention layer.
    It uses weight normalization for linear projections
    (Salimans & Kingma, 2016)  w = g * v/2-norm(v)

    Args:
      in_dim: int last dimension of the inputs
      embed_size: int target embedding size
      layer_id: int the id of current convolution layer
      add_res: bool whether residual connection should be added or not
      mode: str current mode
    """
    super(AttentionLayerNormalized, self).__init__()

    self.add_res = add_res
    self.scaling_factor = scaling_factor
    self.regularizer = regularizer

    with tf.variable_scope("attention_layer_" + str(layer_id)):

      # linear projection layer to project the attention input to target space
      self.tgt_embed_proj = FeedFowardNetworkNormalized(
          in_dim,
          embed_size,
          dropout=1.0,
          var_scope_name="att_linear_mapping_tgt_embed",
          mode=mode,
          normalization_type=normalization_type,
          regularizer=self.regularizer,
          init_var=init_var
      )

      # linear projection layer to project back to the input space
      self.out_proj = FeedFowardNetworkNormalized(
          embed_size,
          in_dim,
          dropout=1.0,
          var_scope_name="att_linear_mapping_out",
          mode=mode,
          normalization_type=normalization_type,
          regularizer=self.regularizer,
          init_var=init_var
      )

  def call(self, input, target_embed, encoder_output_a, encoder_output_b,
           input_attention_bias):
    """Calculates the attention vectors.

    Args:
      input: A float32 tensor with shape [batch_size, length, in_dim]
      target_embed: A float32 tensor with shape [batch_size, length, in_dim]
                    containing the target embeddings
      encoder_output_a: A float32 tensor with shape [batch_size, length, out_dim]
                        containing the first encoder outputs, uses as the keys
      encoder_output_b: A float32 tensor with shape [batch_size, length, src_emb_dim]
                        containing the second encoder outputs, uses as the values
      input_attention_bias: A float32 tensor with shape [batch_size, length, 1]
                            containing the bias used to mask the paddings

    Returns:
      float32 tensor with shape [batch_size, length, out_dim].
    """

    h_proj = self.tgt_embed_proj(input)
    d_proj = (h_proj + target_embed) * self.scaling_factor
    att_score = tf.matmul(d_proj, encoder_output_a, transpose_b=True)

    # Masking need to be done in float32. Added to support mixed-precision training.
    att_score = tf.cast(x=att_score, dtype=tf.float32)

    # mask out the paddings
    if input_attention_bias is not None:
      att_score = att_score + input_attention_bias

    att_score = tf.nn.softmax(att_score)

    # Cast back to original type
    att_score = tf.cast(x=att_score, dtype=encoder_output_b.dtype)

    length = tf.cast(tf.shape(encoder_output_b), encoder_output_b.dtype)
    output = tf.matmul(att_score, encoder_output_b) * \
             length[1] * tf.cast(tf.sqrt(1.0 / length[1]), dtype=encoder_output_b.dtype)
    output = self.out_proj(output)

    if self.add_res:
      output = (output + input) * self.scaling_factor

    return output
