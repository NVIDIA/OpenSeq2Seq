# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from .decoder import Decoder

class JointCTCAttentionDecoder(Decoder):
  """Joint CTC Attention like decoder.
  """
  @staticmethod
  def get_required_params():
    return dict(Decoder.get_required_params(), **{
        'attn_decoder_params': dict,
        'ctc_decoder_params': dict,
        'beam_search_params': dict,
        'language_model_params': dict,
        'GO_SYMBOL': int,  # symbol id
        'END_SYMBOL': int,  # symbol id
        'tgt_vocab_size': int,
        'ctc_decoder': None,
        'attn_decoder': None,
    })

  @staticmethod
  def get_optional_params():
    return dict(Decoder.get_optional_params(), **{
    })

  def __init__(self, params, model, name='jca_decoder', mode='train'):
    """Initializes RNN decoder with embedding.

    See parent class for arguments description.

    Config parameters:
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

    self.ctc_decoder = self.params['ctc_decoder'](params=self.ctc_params, mode=mode, model=model)
    self.attn_decoder = self.params['attn_decoder'](params=self.attn_params, mode=mode, model=model)


  def _decode(self, input_dict):
    """Decodes representation into data.

    Args:
      input_dict (dict): Python dictionary with inputs to decoder.

    Config parameters:
    """
    
    seq_outputs = self.attn_decoder.decode(input_dict=input_dict)
    ctc_outputs = self.ctc_decoder.decode(input_dict=input_dict)

    return {
    		'outputs' : seq_outputs['outputs'],
    		'seq_outputs' : seq_outputs,
    		'ctc_outputs' : ctc_outputs,
    }
