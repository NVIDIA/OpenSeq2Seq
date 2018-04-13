# Copyright (c) 2018 NVIDIA Corporation
"""
Encoders based on Transformers arch from https://arxiv.org/abs/1706.03762
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf

from .encoder import Encoder
from open_seq2seq.parts.attention import multi_head_attention_fn
from open_seq2seq.parts.common import ffn_and_layer_norm, \
                                      embed_and_maybe_add_position_signal, \
                                      normalize


class TransformerEncoder(Encoder):
  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
      'ffn_inner_dim': int,
      'd_model': int,
      'attention_heads': int,
      'src_vocab_size': int,
      'encoder_layers': int,
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
      'encoder_drop_prob': float,
    })

  def __init__(self, params,
               name="transformer_encoder",
               mode='train'):
    """
    Initializes uni-directional encoder with embeddings
    :param params: dictionary with encoder parameters
    Must define:
      * src_vocab_size - data vocabulary size
      * d_model - size of embedding to use
      * time_major (optional)
      * mode - train or infer
      ... add any cell-specific parameters here as well
    """
    super(TransformerEncoder, self).__init__(
      params, name=name, mode=mode,
    )

    self._drop_prob = self.params.get("encoder_drop_prob", 0.0)
    if self._mode != 'train':
      self._drop_prob = 0.0
    self._batch_size = self.params['batch_size_per_gpu']

  def _encode(self, input_dict):
    ffn_inner_dim = self.params["ffn_inner_dim"]
    d_model = self.params['d_model']
    attention_heads = self.params["attention_heads"]
    enc_emb_w = tf.get_variable(name="EncoderEmbeddingMatrix",
                                shape=[
                                  self.params['src_vocab_size'],
                                  self.params['d_model']])

    embedded_inputs_with_pos, bias = embed_and_maybe_add_position_signal(
      inpt=input_dict['src_inputs'],
      emb_W=enc_emb_w,
      num_timescales=int(d_model/2),
      d_model=d_model, heads=attention_heads)

    x = tf.layers.dropout(inputs=embedded_inputs_with_pos,
                          rate=self._drop_prob,
                          noise_shape=[self._batch_size, 1, d_model])

    for block_ind in range(self.params['encoder_layers']):
      with tf.variable_scope("EncoderBlock_{}".format(block_ind)):
        # self-attention
        with tf.variable_scope("SelfAttention"):
          att_out = multi_head_attention_fn(Q=x, K=x, V=x, d_model=d_model,
                                            h=attention_heads, additional_bias=bias)

          att_out = tf.layers.dropout(inputs=att_out, rate=self._drop_prob,
                                      noise_shape=[self._batch_size, 1, d_model])
          ff_input = normalize(att_out + x)
        x = ffn_and_layer_norm(inpt=ff_input,
                               inner_dim=ffn_inner_dim,
                               resulting_dim=d_model,
                               drop_prob=self._drop_prob)
    return {'encoder_outputs': x,
            'encoder_state': None,
            'src_lengths': input_dict['src_lengths'],
            'enc_emb_w': enc_emb_w,
            'encoder_input': input_dict['src_inputs']}
