# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from open_seq2seq.parts.rnns.attention_wrapper import BahdanauAttention, \
                                                      LuongAttention, \
                                                      AttentionWrapper
from open_seq2seq.parts.rnns.rnn_beam_search_decoder import BeamSearchDecoder
from open_seq2seq.parts.rnns.utils import single_cell
from .decoder import Decoder

class ListenAttendSpellDecoder(Decoder):
  """Listen Attend Spell like decoder.
  """
  @staticmethod
  def get_required_params():
    return dict(Decoder.get_required_params(), **{
        'GO_SYMBOL': int,  # symbol id
        'END_SYMBOL': int,  # symbol id
        'tgt_vocab_size': int,
        'tgt_emb_size': int,
        'attention_layer_size': int,
        'attention_type': ['bahdanau', 'luong'],
        'core_cell': None,
        'decoder_layers': int,
        'decoder_use_skip_connections': bool,
        'batch_size': int,
    })

  @staticmethod
  def get_optional_params():
    return dict(Decoder.get_optional_params(), **{
        'core_cell_params': dict,
        'bahdanau_normalize': bool,
        'luong_scale': bool,
        'decoder_dp_input_keep_prob': float,
        'decoder_dp_output_keep_prob': float,
        'time_major': bool,
        'use_swap_memory': bool,
        'proj_size': int,
        'num_groups': int,
        'PAD_SYMBOL': int,  # symbol id
    })

  def __init__(self, params, model, name='las_decoder', mode='train'):
    """Initializes RNN decoder with embedding.

    See parent class for arguments description.

    Config parameters:
    """
    super(ListenAttendSpellDecoder, self).__init__(params, model, name, mode)
    self._batch_size = self.params['batch_size']
    self.GO_SYMBOL = self.params['GO_SYMBOL']
    self.END_SYMBOL = self.params['END_SYMBOL']
    self._tgt_vocab_size = self.params['tgt_vocab_size']
    self._tgt_emb_size = self.params['tgt_emb_size']

  def _decode(self, input_dict):
    """Decodes representation into data.

    Args:
      input_dict (dict): Python dictionary with inputs to decoder.

    Config parameters:
    """
    encoder_outputs = input_dict['encoder_output']['outputs']
    enc_src_lengths = input_dict['encoder_output']['src_lengths']
    tgt_inputs = input_dict['target_tensors'][0] if 'target_tensors' in \
                                                    input_dict else None
    tgt_lengths = input_dict['target_tensors'][1] if 'target_tensors' in \
                                                     input_dict else None

    self._target_emb_layer = tf.get_variable(
        name='TargetEmbeddingMatrix',
        shape=[self._tgt_vocab_size, self._tgt_emb_size],
        dtype=tf.float32,
    )

    self._output_projection_layer = tf.layers.Dense(
        self._tgt_vocab_size, use_bias=False,
    )


