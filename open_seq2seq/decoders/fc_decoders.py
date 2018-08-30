# Copyright (c) 2018 NVIDIA Corporation
"""This module defines various fully-connected decoders (consisting of one
fully connected layer).

These classes are usually used for models that are not really
sequence-to-sequence and thus should be artificially split into encoder and
decoder by cutting, for example, on the last fully-connected layer.
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import os

import tensorflow as tf

from .decoder import Decoder


class FullyConnectedDecoder(Decoder):
  """Simple decoder consisting of one fully-connected layer.
  """
  @staticmethod
  def get_required_params():
    return dict(Decoder.get_required_params(), **{
        'output_dim': int,
    })

  def __init__(self, params, model,
               name="fully_connected_decoder", mode='train'):
    """Fully connected decoder constructor.

    See parent class for arguments description.

    Config parameters:

    * **output_dim** (int) --- output dimension.
    """
    super(FullyConnectedDecoder, self).__init__(params, model, name, mode)

  def _decode(self, input_dict):
    """This method performs linear transformation of input.

    Args:
      input_dict (dict): input dictionary that has to contain
          the following fields::
            input_dict = {
              'encoder_output': {
                'outputs': output of encoder (shape=[batch_size, num_features])
              }
            }

    Returns:
      dict: dictionary with the following tensors::

        {
          'logits': logits with the shape=[batch_size, output_dim]
          'outputs': [logits] (same as logits but wrapped in list)
        }
    """
    inputs = input_dict['encoder_output']['outputs']
    regularizer = self.params.get('regularizer', None)

    # activation is linear by default
    logits = tf.layers.dense(
        inputs=inputs,
        units=self.params['output_dim'],
        kernel_regularizer=regularizer,
        name='fully_connected',
    )
    return {'logits': logits, 'outputs': [logits]}


class FullyConnectedTimeDecoder(Decoder):
  """Fully connected decoder that operates on inputs with time dimension.
  That is, input shape should be ``[batch size, time length, num features]``.
  """
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
    """Fully connected time decoder constructor.

    See parent class for arguments description.

    Config parameters:

    * **tgt_vocab_size** (int) --- target vocabulary size, i.e. number of
      output features.
    * **logits_to_outputs_func** --- function that maps produced logits to
      decoder outputs, i.e. actual text sequences.
    """
    super(FullyConnectedTimeDecoder, self).__init__(params, model, name, mode)

  def _decode(self, input_dict):
    """Creates TensorFlow graph for fully connected time decoder.

    Args:
      input_dict (dict): input dictionary that has to contain
          the following fields::
            input_dict = {
              'encoder_output': {
                "outputs": tensor with shape [batch_size, time length, hidden dim]
                "src_length": tensor with shape [batch_size]
              }
            }

    Returns:
      dict: dictionary with the following tensors::

        {
          'logits': logits with the shape=[time length, batch_size, tgt_vocab_size]
          'outputs': logits_to_outputs_func(logits, input_dict)
        }
    """
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
      outputs = self.params['logits_to_outputs_func'](logits, input_dict)
      return {
          'outputs': outputs,
          'logits': logits,
          'src_length': input_dict['encoder_output']['src_length'],
      }
    return {'logits': logits,
            'src_length': input_dict['encoder_output']['src_length']}


class FullyConnectedCTCDecoder(FullyConnectedTimeDecoder):
  """Fully connected time decoder that provides a CTC-based text
  generation (either with or without language model). If language model is not
  used, ``tf.nn.ctc_greedy_decoder`` will be used as text generation method.
  """
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
        'alpha': float,
        'beta': float,
        'lm_path': str,
        'trie_path': str,
        'alphabet_config_path': str,
    })

  def __init__(self, params, model,
               name="fully_connected_ctc_decoder", mode='train'):
    """Fully connected CTC decoder constructor.

    See parent class for arguments description.

    Config parameters:

    * **use_language_model** (bool) --- whether to use language model for
      output text generation. If False, other config parameters are not used.
    * **decoder_library_path** (string) --- path to the ctc decoder with
      language model library.
    * **lm_path** (string) --- path to the language model file.
    * **trie_path** (string) --- path to the prefix trie file.
    * **alphabet_config_path** (string) --- path to the alphabet file.
    * **beam_width** (int) --- beam width for beam search.
    * **alpha** (float) --- weight that is assigned to language model
      probabilities.
    * **beta** (float) --- weight that is assigned to the
      word count.
    """
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
                model_path=self.params['lm_path'], trie_path=self.params['trie_path'],
                alphabet_path=self.params['alphabet_config_path'],
                alpha=self.params['alpha'],
                beta=self.params['beta'],
                top_paths=top_paths, merge_repeated=merge_repeated,
            )
        )
        return [tf.SparseTensor(decoded_ixs[0], decoded_vals[0],
                                decoded_shapes[0])]

      self.params['logits_to_outputs_func'] = decode_with_lm
    else:
      def decode_without_lm(logits, decoder_input, merge_repeated=True):
        if logits.dtype.base_dtype != tf.float32:
          logits = tf.cast(logits, tf.float32)
        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
            logits, decoder_input['encoder_output']['src_length'],
            merge_repeated,
        )
        return decoded

      self.params['logits_to_outputs_func'] = decode_without_lm
