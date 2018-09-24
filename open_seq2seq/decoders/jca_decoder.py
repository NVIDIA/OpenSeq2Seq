# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from .decoder import Decoder


class JointCTCAttentionDecoder(Decoder):
  """Joint CTC Attention like decoder.
  Combines CTC and Attention based decoder.
  Use only outputs from the Attention decoder during inference.
  """
  @staticmethod
  def get_required_params():
    return dict(Decoder.get_required_params(), **{
        'ctc_decoder': None,
        'attn_decoder': None,
        'attn_decoder_params': dict,
        'ctc_decoder_params': dict,
        'beam_search_params': dict,
        'language_model_params': dict,
        'GO_SYMBOL': int,  # symbol id
        'END_SYMBOL': int,  # symbol id
        'tgt_vocab_size': int,
    })

  @staticmethod
  def get_optional_params():
    return dict(Decoder.get_optional_params(), **{
    })

  def __init__(self, params, model, name='jca_decoder', mode='train'):
    """Initializes RNN decoder with embedding.

    See parent class for arguments description.

    Config parameters:

    * **ctc_decoder** (any class derived from
      :class:`Decoder <decoders.decoder.Decoder>`) --- CTC decoder class to use.
    * **attn_decoder** (any class derived from
      :class:`Decoder <decoders.decoder.Decoder>`) --- Attention decoder class to use.
    * **attn_decoder_params** (dict) --- parameters for the attention decoder.
    * **ctc_decoder_params** (dict) --- parameters for the ctc decoder.
    * **beam_search_params** (dict) --- beam search parameters for decoding using the attention based decoder.
    * **language_model_params** (dict) --- language model parameters for decoding with an external language model.

    * **GO_SYMBOL** (int) --- GO symbol id, must be the same as used in
      data layer.
    * **END_SYMBOL** (int) --- END symbol id, must be the same as used in
      data layer.
    * **tgt_vocab_size** (int) --- vocabulary size of the targets to use for final softmax.
    """
    super(JointCTCAttentionDecoder, self).__init__(params, model, name, mode)

    self.ctc_params = self.params['ctc_decoder_params']
    self.attn_params = self.params['attn_decoder_params']
    self.beam_search_params = self.params['beam_search_params']
    self.lang_model_params = self.params['language_model_params']

    self.attn_params.update(self.beam_search_params)
    self.attn_params.update(self.lang_model_params)

    self.ctc_params['tgt_vocab_size'] = self.params['tgt_vocab_size'] - 1
    self.attn_params['tgt_vocab_size'] = self.params['tgt_vocab_size']
    self.attn_params['GO_SYMBOL'] = self.params['GO_SYMBOL']
    self.attn_params['END_SYMBOL'] = self.params['END_SYMBOL']

    self.ctc_decoder = self.params['ctc_decoder'](
        params=self.ctc_params, mode=mode, model=model)
    self.attn_decoder = self.params['attn_decoder'](
        params=self.attn_params, mode=mode, model=model)

  def _decode(self, input_dict):
    """Joint decoder that combines Attention and CTC outputs.

    Args:
      input_dict (dict): Python dictionary with inputs to decoder.

    Config parameters:

    * **src_inputs** --- Decoder input Tensor of shape [batch_size, time, dim]
    * **src_lengths** --- Decoder input lengths Tensor of shape [batch_size]
    * **tgt_inputs** --- Only during training. labels Tensor of the
      shape [batch_size, time].
    * **tgt_lengths** --- Only during training. label lengths
      Tensor of the shape [batch_size].

    Returns:
      dict: Python dictionary with:
      * outputs - tensor of shape [batch_size, time] from the Attention decoder
      * seq_outputs - output dictionary from the Attention decoder
      * ctc_outputs - output dictionary from the CTC decoder
    """

    seq_outputs = self.attn_decoder.decode(input_dict=input_dict)
    ctc_outputs = self.ctc_decoder.decode(input_dict=input_dict)

    return {
        'outputs': seq_outputs['outputs'],
        'seq_outputs': seq_outputs,
        'ctc_outputs': ctc_outputs,
    }
