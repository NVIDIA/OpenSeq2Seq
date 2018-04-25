# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os

from .decoder import Decoder


class FullyConnectedTimeDecoder(Decoder):
  @staticmethod
  def get_required_params():
    return dict(Decoder.get_required_params(), **{
      'tgt_vocab_size': int,
    })

  @staticmethod
  def get_optional_params():
    return dict(Decoder.get_optional_params(), **{
      'logits_to_outputs_func': None,  # user defined function
    })

  def __init__(self, params, model,
               name="fully_connected_time_decoder", mode='train'):
    super(FullyConnectedTimeDecoder, self).__init__(params, model, name, mode)

  def _decode(self, input_dict):
    inputs = input_dict['encoder_output']['outputs']
    regularizer = self.params.get('regularizer', None)

    batch_size, _, n_hidden = inputs.get_shape().as_list()
    # reshape from [B, T, A] --> [B*T, A].
    # Output shape: [n_steps * batch_size, n_hidden]
    inputs = tf.reshape(inputs, [-1, n_hidden])

    # activation is linear by default
    logits = tf.layers.dense(
      inputs=inputs,
      units=self.params['tgt_vocab_size'],
      kernel_regularizer=regularizer,
      name='fully_connected',
    )
    logits = tf.reshape(
      logits,
      [batch_size, -1, self.params['tgt_vocab_size']],
      name="logits",
    )
    # converting to time_major=True shape
    logits = tf.transpose(logits, [1, 0, 2])

    if 'logits_to_outputs_func' in self.params:
      samples = self.params['logits_to_outputs_func'](logits, input_dict)
      return {
        'samples': samples,
        'logits': logits,
        'src_length': input_dict['encoder_output']['src_length'],
      }
    return {'logits': logits,
            'src_length': input_dict['encoder_output']['src_length']}


class FullyConnectedCTCDecoder(FullyConnectedTimeDecoder):
  @staticmethod
  def get_required_params():
    return dict(FullyConnectedTimeDecoder.get_required_params(), **{
      'use_language_model': bool,
    })

  @staticmethod
  def get_optional_params():
    return dict(FullyConnectedTimeDecoder.get_optional_params(), **{
      'decoder_library_path': str,
      'beam_width': int,
      'lm_weight': float,
      'word_count_weight': float,
      'valid_word_count_weight': float,
      'lm_binary_path': str,
      'lm_trie_path': str,
      'alphabet_config_path': str,
    })

  def __init__(self, params, model,
               name="fully_connected_ctc_decoder", mode='train'):
    super(FullyConnectedCTCDecoder, self).__init__(params, model, name, mode)

    if self.params['use_language_model']:
      # creating decode_with_lm function if it is compiled
      lib_path = self.params['decoder_library_path']
      if not os.path.exists(os.path.abspath(lib_path)):
        raise IOError('Can\'t find the decoder with language model library. '
                      'Make sure you have built it and '
                      'check that you provide the correct '
                      'path in the --decoder_library_path parameter.')

      custom_op_module = tf.load_op_library(lib_path)

      def decode_with_lm(logits, decoder_input,
                         beam_width=self.params['beam_width'],
                         top_paths=1, merge_repeated=False):
        sequence_length = decoder_input['encoder_output']['src_length']
        if logits.dtype.base_dtype != tf.float32:
          logits = tf.cast(logits, tf.float32)
        decoded_ixs, decoded_vals, decoded_shapes, log_probabilities = (
          custom_op_module.ctc_beam_search_decoder_with_lm(
            logits, sequence_length, beam_width=beam_width,
            model_path=self.params['lm_binary_path'],
            trie_path=self.params['lm_trie_path'],
            alphabet_path=self.params['alphabet_config_path'],
            lm_weight=self.params['lm_weight'],
            word_count_weight=self.params['word_count_weight'],
            valid_word_count_weight=self.params['valid_word_count_weight'],
            top_paths=top_paths, merge_repeated=merge_repeated,
          )
        )
        return tf.SparseTensor(decoded_ixs[0], decoded_vals[0], decoded_shapes[0])

      self.params['logits_to_outputs_func'] = decode_with_lm
    else:
      def decode_without_lm(logits, decoder_input, merge_repeated=True):
        if logits.dtype.base_dtype != tf.float32:
          logits = tf.cast(logits, tf.float32)
        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
          logits, decoder_input['encoder_output']['src_length'],
          merge_repeated,
        )
        return decoded[0]

      self.params['logits_to_outputs_func'] = decode_without_lm
