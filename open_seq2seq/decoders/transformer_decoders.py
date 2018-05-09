# Copyright (c) 2018 NVIDIA Corporation
"""
Encoders based on Transformers arch from https://arxiv.org/abs/1706.03762
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
from .decoder import Decoder

from open_seq2seq.parts.attention import multi_head_attention_fn
from open_seq2seq.parts.common import ffn_and_layer_norm, \
                                      embed_and_maybe_add_position_signal, \
                                      get_pad_masking_bias, \
                                      dropout_normalize_add_NTC
from open_seq2seq.data.text2text import SpecialTextTokens


def transformer_decoder_fn(decoder_input_seq,
                           encoder_input_seq,
                           encoder_outputs,
                           dec_emb_w,
                           output_projector,
                           num_decoder_blocks,
                           d_model,
                           ffn_inner_dim,
                           attention_heads,
                           norm_type,
                           drop_prob=0.0,
                           training=True,
                           ):
  """
  Transformer decoder function. It will train all steps in parallel,
  but for inference should be used in auto-regressive manner.
  :param decoder_input_seq: input sequence to the decoder before the embedding
  of shape [batch_size, T]
  For iteration 0 during inference, this should contain GO symbols and be of
  shape [batch_size, 1]
  :param encoder_input_seq: input sequence to the encoder before the embedding
  of shape [batch_size, T]. Necessary for pad masking biases
  :param encoder_outputs: representation computed by encoder,
  tensor of shape [batch, T, dim]
  :param dec_emb_w: embedding matrix for decoder inputs
  :param output_projector: output projection layer for decoder
  :param num_decoder_blocks: number of decoder layers/blocks
  :param d_model: Int
  :param ffn_inner_dim: Int - dimensionality for the middle for ffn transorm
  :param attention_heads: Int - number of attention heads
  :param dropout_drop_prob: (default: 0.0) dropout drop probability
  :return: Logits tensor of shape [batch, time, d_model]
  """

  with tf.variable_scope("transformer_decoder_fn"):

    x, decoder_self_bias = embed_and_maybe_add_position_signal(
      inpt=decoder_input_seq,
      emb_W=dec_emb_w,
      num_timescales=int(d_model/2),
      heads=attention_heads)      
   
    encoder_decoder_bias = get_pad_masking_bias(x=decoder_input_seq,
                                                y=encoder_input_seq,
                                                PAD_ID=SpecialTextTokens.PAD_ID.value,
                                                heads=attention_heads,
                                                dtype=dec_emb_w.dtype)

    x = dropout_normalize_add_NTC(x=x,
                                  drop_prob=drop_prob,
                                  training=training,
                                  norm_type=norm_type)

    for block_ind in range(num_decoder_blocks):
      with tf.variable_scope("DecoderBlock_{}".format(block_ind)):
        with tf.variable_scope("SelfAttention"):
          att_out = multi_head_attention_fn(Q=x, K=x, V=x,
                                            d_model=d_model,
                                            # because of residual connections this is masked in every block
                                            mask_future=True,
                                            h=attention_heads,
                                            additional_bias=decoder_self_bias)

          x = dropout_normalize_add_NTC(x=att_out, residual_x=x,
                                        drop_prob=drop_prob,
                                        training=training,
                                        norm_type=norm_type)
        with tf.variable_scope("Attend2Encoder"):
          att_out = multi_head_attention_fn(Q=x,
                                            K=encoder_outputs,
                                            V=encoder_outputs,
                                            d_model=d_model,
                                            h=attention_heads,
                                            additional_bias=encoder_decoder_bias)
          x = dropout_normalize_add_NTC(x=att_out, residual_x=x,
                                        drop_prob=drop_prob,
                                        training=training,
                                        norm_type=norm_type)

        x = ffn_and_layer_norm(x,
                               inner_dim=ffn_inner_dim,
                               resulting_dim=d_model,
                               drop_prob=drop_prob,
                               training=training,
                               norm_type=norm_type,
                               )
    
    result = output_projector(x)    
    return result


class TransformerDecoder(Decoder):
  """Greedy Transformer Decoder
  """
  def __init__(self, params, model,
               name="transformer_decoder", mode='train'):
    """
    Initializes Decoder
    :param params: dictionary of decoder parameters
    """
    super(TransformerDecoder, self).__init__(params, model, name, mode)
    self._batch_size = self.params['batch_size']
    self.GO_SYMBOL = self.params['GO_SYMBOL']
    self.END_SYMBOL = self.params['END_SYMBOL']
    self._tgt_vocab_size = self.params['tgt_vocab_size']
    self._tgt_emb_size = self.params['d_model']
    self._norm_type = self.params.get("decoder_norm_type", 'layer_norm')
    self._drop_prob = self.params.get("decoder_drop_prob", 0.0)
    if self._mode != 'train':
      self._drop_prob = 0.0
    self._is_unittest = False

  @staticmethod
  def get_required_params():
    return dict(Decoder.get_required_params(), **{
      "initializer": None,
      "d_model": int,
      "ffn_inner_dim": int,
      "decoder_layers": int,
      "attention_heads": int,
      "GO_SYMBOL": int,
      "END_SYMBOL": int,
      "PAD_SYMBOL": int,
      "tgt_vocab_size": int,
      'batch_size': int,
    })

  @staticmethod
  def get_optional_params():
    return dict(Decoder.get_optional_params(), **{
      "use_encoder_emb": bool,
      "tie_emb_and_proj": bool,
      "decoder_drop_prob": float,
      "decoder_norm_type": str,
    })

  def _decode(self, input_dict):
    """
    Decodes decoder input sequence.
    :param input_dict: dictionary of decoder inputs
    For example (but may differ):
    decoder_input= { "src_inputs" : decoder source sequence,
                     "src_lengths" : decoder source length,
                     "tgt_inputs" :  (during training),
                     "tgt_lengths" : (during training)}

    :return: dictionary of decoder outputs
    For example (but may differ):
    decoder_output = {"decoder_outputs" : decoder_outputs,
                      "decoder_lengths" : decoder_lengths}
    """
    encoder_outputs = input_dict['encoder_output']['outputs']
    encoder_input = input_dict['encoder_output']['encoder_input']
    ffn_inner_dim = self.params["ffn_inner_dim"]
    d_model = self.params['d_model']
    attention_heads = self.params["attention_heads"]

    with tf.variable_scope("TransformerDecoder"):
      if self.params.get('use_encoder_emb', False):
        dec_emb_w = input_dict['encoder_output']['enc_emb_w']
      else:
        dec_emb_w = tf.get_variable(name='DecoderEmbeddingMatrix',
                                    shape=[self._tgt_vocab_size,
                                           self._tgt_emb_size],
                                    dtype=self.params['dtype'])
      if self.params.get('tie_emb_and_proj', False):
        output_projection_layer = lambda x: tf.reshape(
          tf.matmul(a=tf.reshape(x, shape=[-1, self._tgt_emb_size]),
                    b=dec_emb_w, transpose_b=True, name="DecoderOutProjection"),
          shape=[self._batch_size, -1, self._tgt_vocab_size])
      else:
        output_projection_layer = tf.layers.Dense(
          units=self._tgt_vocab_size,
          use_bias=False,
          name="DecoderOutProjection")

      tgt_inputs = input_dict['tgt_sequence']

      if self._mode == 'train':
        training = True
        drop_prob = self._drop_prob

        output = transformer_decoder_fn(
          decoder_input_seq=tgt_inputs,
          encoder_input_seq=encoder_input,
          encoder_outputs=encoder_outputs,
          dec_emb_w=dec_emb_w,
          output_projector=output_projection_layer,
          num_decoder_blocks=self.params["decoder_layers"],
          d_model=d_model,
          ffn_inner_dim=ffn_inner_dim,
          attention_heads=attention_heads,
          drop_prob=drop_prob,
          training=training,
          norm_type=self._norm_type,
        )

        return {
          "logits": output,
          "samples": tf.argmax(output, axis=-1),
          "final_state": None,
          "final_sequence_lengths": None}

      else:# Decoder must be used in auto-regressive manner
        training = False
        drop_prob = 0.0
        decoding_length = encoder_outputs.shape[1].value or \
                          tf.shape(encoder_outputs)[1]

        decoder_ids_so_far = tf.fill([self._batch_size, 1],
                                     value=self.GO_SYMBOL)
        done_decoding = tf.fill([self._batch_size], False)
        i = tf.constant(0)

        def loop_stop_condition(loop_var, is_done, *_):
          return ((loop_var < decoding_length) &
                  tf.logical_not(tf.reduce_all(is_done)))

        def while_function(loop_var, _done_decoding, steps_so_far, _output):
          step_logits = transformer_decoder_fn(
            decoder_input_seq=steps_so_far if not self._is_unittest else
            tgt_inputs[:, :loop_var+1],
            encoder_input_seq=encoder_input,
            encoder_outputs=encoder_outputs,
            dec_emb_w=dec_emb_w,
            output_projector=output_projection_layer,
            num_decoder_blocks=self.params["decoder_layers"],
            d_model=d_model,
            ffn_inner_dim=ffn_inner_dim,
            attention_heads=attention_heads,
            drop_prob=drop_prob,
            training=training,
            norm_type=self._norm_type,
          )

          decoder_argmx = tf.argmax(step_logits, axis=-1, output_type=tf.int32)
          step_out = decoder_argmx[:, -1] # this is of shape [batch, T]
          if not self._is_unittest:
            _done_decoding |= tf.equal(step_out, self.END_SYMBOL)

          steps_so_far = tf.concat([steps_so_far, tf.expand_dims(step_out, 1)],
                                   axis=1)
          return tf.add(loop_var, 1), _done_decoding, steps_so_far, step_logits
        
        output = tf.zeros(shape=[self._batch_size, decoding_length,
                                 self._tgt_vocab_size],
                          dtype=self.params['dtype'])

        _, _, decoder_ids_so_far, output = tf.while_loop(cond=loop_stop_condition,
                                                         body=while_function,
                                                         loop_vars=[i, done_decoding, decoder_ids_so_far, output],
                                                         parallel_iterations=1,
                                                         shape_invariants=[i.get_shape(),
                                                                           done_decoding.get_shape(),
                                                                           tf.TensorShape([self._batch_size, None]),
                                                                           tf.TensorShape([None, None, None])])
        
        return {"logits": output,
                "samples": [decoder_ids_so_far[:, 1:]],
                "final_state": None,
                "final_sequence_lengths": None}

  @property
  def params(self):
    """Parameters used to construct the encoder"""
    return self._params
