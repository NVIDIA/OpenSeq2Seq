# Copyright (c) 2018 NVIDIA Corporation
"""
RNN-based decoders
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import copy
import tensorflow as tf

from open_seq2seq.parts.rnns.gnmt import GNMTAttentionMultiCell, \
                                                           gnmt_residual_fn
from open_seq2seq.parts.rnns.utils import create_rnn_cell
from open_seq2seq.parts.rnns.attention_wrapper import BahdanauAttention, \
                                                 LuongAttention, \
                                                 LocationSensitiveAttention, \
                                                 AttentionWrapper
from .decoder import Decoder
from open_seq2seq.parts.rnns.rnn_beam_search_decoder import BeamSearchDecoder
from open_seq2seq.parts.tacotron.tacotron_helper import TacotronHelper, TacotronTrainingHelper, TrainingHelper, InferenceHelper, ScheduledSamplingHelper
from open_seq2seq.parts.tacotron.tacotron_decoder import TacotronDecoder
# from open_seq2seq.parts.tacotron.decoder import dynamic_decode
from tensorflow.contrib.rnn import LSTMStateTuple



def conv1d_bn_actv(name, inputs, filters, kernel_size, activation_fn, strides,
                   padding, regularizer, training, use_bias, data_format, 
                   enable_bn, bn_momentum, bn_epsilon):
  """Helper function that applies 1-D convolution, batch norm and activation."""
  conv = tf.layers.conv1d(
    name="{}".format(name),
    inputs=inputs,
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding=padding,
    kernel_regularizer=regularizer,
    use_bias=use_bias,
    data_format=data_format,
  )
  output = conv
  if enable_bn:
    bn = tf.layers.batch_normalization(
      name="{}/bn".format(name),
      inputs=conv,
      gamma_regularizer=regularizer,
      training=training,
      axis=-1 if data_format == 'channels_last' else 1,
      momentum=bn_momentum,
      epsilon=bn_epsilon,
    )
    output = bn
  if activation_fn is not None:
    output = activation_fn(output)
  return output

class Tacotron2Decoder(Decoder):
  """
  Typical RNN decoder with attention mechanism
  """
  @staticmethod
  def get_required_params():
    return dict(Decoder.get_required_params(), **{
      'attention_layer_size': int,
      'attention_type': ['bahdanau', 'luong', 'gnmt', 'gnmt_v2', 'location', None],
      'attention_rnn_enable': bool,
      'decoder_cell_units': int,
      'decoder_cell_type': ['lstm', 'gru', 'glstm', 'slstm'],
      'decoder_layers': int,
      'num_audio_features': int,
      'scheduled_sampling_prob': float,
    })

  @staticmethod
  def get_optional_params():
    return dict(Decoder.get_optional_params(), **{
      'attention_rnn_units': int,
      'attention_rnn_layers': int,
      'attention_rnn_cell_type': ['lstm', 'gru', 'glstm', 'slstm'],
      'bahdanau_normalize': bool,
      'luong_scale': bool,
      'decoder_use_skip_connections': bool,
      'decoder_dp_input_keep_prob': float,
      'decoder_dp_output_keep_prob': float,
      'time_major': bool,
      'use_swap_memory': bool,
      'enable_prenet': bool,
      'prenet_layers': int,
      'prenet_units': int,
      'prenet_activation': None,
      'enable_postnet': bool,
      'postnet_conv_layers': list,
      'postnet_enable_bn': bool,
      'postnet_use_bias': bool,
      'postnet_bn_momentum': float,
      'postnet_bn_epsilon': float,
      'postnet_data_format': ['channels_first', 'channels_last'],
      'postnet_keep_dropout_prob': float,
      "anneal_sampling_prob": bool,
      "sampling_test": bool,
      "mask_decoder_sequence": bool,
      "use_prenet_output": bool,
      "attention_bias": bool,
    })

  def __init__(self, params, model,
               name='tacotron_2_decoder', mode='train'):
    """Tacotron-2 like decoder constructor.

    See parent class for arguments description.

    Config parameters:

    * **attention_layer_size** (int) --- size of attention layer.
    * **attention_type** (string) --- Determines whether attention mechanism to use,
      should be one of 'bahdanau', 'luong', 'gnmt', 'gnmt_v2', 'location', or None.
      Use of 'location'-sensitive attention is strongly recommended.
    * **attention_rnn_enable** (bool) --- Whether to create a rnn layer for the
      attention mechanism. If false, the attention mechanism is wrapped around the
      decoder rnn
    * **attention_rnn_units** (int) --- dimension of attention RNN cells if enabled.
      Defaults to 1024.
    * **attention_rnn_layers** (int) --- number of attention RNN layers to use if enabled.
      Defaults to 1.
    * **attention_rnn_cell_type** (string) --- could be "lstm", "gru", "glstm", or "slstm".
      Currently, only 'lstm' has been tested. Defaults to 'lstm'.
    * **bahdanau_normalize** (bool) ---  Defaults to False.
    * **luong_scale** (bool) ---  Defaults to False.
    * **decoder_rnn_units** (int) --- dimension of decoder RNN cells.
    * **decoder_rnn_layers** (int) --- number of decoder RNN layers to use.
    * **decoder_rnn_cell_type** (string) --- could be "lstm", "gru", "glstm", or "slstm".
    * **decoder_use_skip_connections** (bool) --- whether to use residual connections in the
      decoder rnn. Defaults to False
    * **decoder_dp_input_keep_prob** (float)
    * **decoder_dp_output_keep_prob** (float)
    * **scheduled_sampling_prob** (float) --- probability for scheduled sampling. Set to 0 for
      teacher forcing.
    * **time_major** (bool) --- whether to output as time major or batch major. Default is False
      for batch major.
    * **use_swap_memory** (bool) --- default is False.
    * **enable_prenet** (bool) --- whether to use the fully-connected prenet in the decoder.
      Defaults to True
    * **prenet_layers** (int) --- number of fully-connected layers to use. Defaults to 2.
    * **prenet_units** (int) --- number of units in each layer. Defaults to 256.
    * **enable_postnet** (bool) --- whether to use the convolutional postnet in the decoder.
      Defaults to True
    * **postnet_conv_layers** (bool) --- list with the description of convolutional
      layers. Must be passed if postnet is enabled
      For example::
        "postnet_conv_layers": [
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME",
            "activation_fn": tf.nn.tanh
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME",
            "activation_fn": tf.nn.tanh
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME",
            "activation_fn": tf.nn.tanh
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME",
            "activation_fn": tf.nn.tanh
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 80, "padding": "SAME",
            "activation_fn": None
          }
        ]
    * **postnet_bn_momentum** (float) --- momentum for batch norm. Defaults to 0.1.
    * **postnet_bn_epsilon** (float) --- epsilon for batch norm. Defaults to 1e-5.
    * **postnet_enable_bn** (bool) --- whether to enable batch norm after each postnet conv layer.
      Defaults to True
    * **postnet_use_bias** (bool) --- whether to enable a bias unit for the postnet conv layers
      Defaults to True
    * **postnet_data_format** (string) --- could be either "channels_first" or
      "channels_last". Defaults to "channels_last".
    * **postnet_keep_dropout_prob** (flaot) --- keep probability for dropout in the postnet conv layers.
      Default to 0.5.
    """

    super(Tacotron2Decoder, self).__init__(params, model, name, mode)
    self.num_audio_features = self.params['num_audio_features']
    self.model = model

  def _build_attention(self,
                       encoder_outputs,
                       encoder_sequence_length,
                       attention_bias):
    """
    Builds Attention part of the graph.
    Currently supports "bahdanau", "luong", and "location"
    :param encoder_outputs:
    :param encoder_sequence_length:
    :return:
    """
    with tf.variable_scope("AttentionMechanism"):
      attention_depth = self.params['attention_layer_size']
      if self.params['attention_type'] == 'location':
        attention_mechanism = LocationSensitiveAttention(
          num_units=attention_depth,
          memory=encoder_outputs,
          memory_sequence_length=encoder_sequence_length,
          probability_fn=tf.nn.softmax,
          dtype=tf.get_variable_scope().dtype,
          use_bias=attention_bias
        )
      elif self.params['attention_type'] == 'bahdanau':
        bah_normalize = self.params.get('bahdanau_normalize', False)
        # attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        attention_mechanism = BahdanauAttention(
          num_units=attention_depth,
          memory=encoder_outputs,
          normalize=bah_normalize,
          memory_sequence_length=encoder_sequence_length,
          probability_fn=tf.nn.softmax,
          dtype=tf.get_variable_scope().dtype
        )
      elif self.params['attention_type'] == 'luong':
        luong_scale = self.params.get('luong_scale', False)
        # attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        attention_mechanism = LuongAttention(
          num_units=attention_depth,
          memory=encoder_outputs,
          scale=luong_scale,
          memory_sequence_length=encoder_sequence_length,
          probability_fn=tf.nn.softmax,
          dtype=tf.get_variable_scope().dtype
        )
      elif self.params['attention_type'] == 'gnmt' or \
           self.params['attention_type'] == 'gnmt_v2':
        # attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        attention_mechanism = BahdanauAttention(
          num_units=attention_depth,
          memory=encoder_outputs,
          normalize=True,
          memory_sequence_length=encoder_sequence_length,
          probability_fn=tf.nn.softmax,
          dtype=tf.get_variable_scope().dtype
        )
      else:
        raise ValueError('Unknown Attention Type')
      return attention_mechanism

  @staticmethod
  def _add_residual_wrapper(cells, start_ind=1):
    for idx, cell in enumerate(cells):
      if idx >= start_ind:
        cells[idx] = tf.contrib.rnn.ResidualWrapper(
          cell,
          residual_fn=gnmt_residual_fn,
        )
    return cells

  def _decode(self, input_dict):
    """
    Decodes representation into data
    :param input_dict: Python dictionary with inputs to decoder
    Must define:
      * src_inputs - decoder input Tensor of shape [batch_size, time, dim]
                     or [time, batch_size, dim]
      * src_lengths - decoder input lengths Tensor of shape [batch_size]
      * tgt_inputs - Only during training. labels Tensor of the
                     shape [batch_size, time] or [time, batch_size]
      * tgt_lengths - Only during training. labels lengths
                      Tensor of the shape [batch_size]
    :return: a Python dictionary with:
      * final_outputs - tensor of shape [batch_size, time, dim]
                        or [time, batch_size, dim]
      * final_state - tensor with decoder final state
      * final_sequence_lengths - tensor of shape [batch_size, time]
                                 or [time, batch_size]
    """
    encoder_outputs = input_dict['encoder_output']['outputs']
    enc_src_lengths = input_dict['encoder_output']['src_length']
    # enc_state = input_dict['encoder_output']['state']
    if self._mode != "infer":
      spec = input_dict['target_tensors'][0] if 'target_tensors' in \
                                                    input_dict else None
      target = input_dict['target_tensors'][1] if 'target_tensors' in \
                                                    input_dict else None
      spec_length = input_dict['target_tensors'][2] if 'target_tensors' in \
                                                    input_dict else None
    _batch_size = encoder_outputs.get_shape().as_list()[0]
    # mean_pool = tf.reduce_mean(encoder_outputs, axis=1)

    training = (self._mode == "train")
    regularizer = self.params.get('regularizer', None)
    bn_momentum = self.params.get('postnet_bn_momentum', 0.1)
    bn_epsilon = self.params.get('postnet_bn_epsilon', 1e-5)
    data_format = self.params.get('postnet_data_format', 'channels_last')
    enable_prenet = self.params.get('enable_prenet', True)
    prenet_layers = self.params.get('prenet_layers', True)
    prenet_units = self.params.get('prenet_units', True)
    
    if self.params.get('enable_postnet', True):
      if "postnet_conv_layers" not in self.params:
        raise ValueError(
            "postnet_conv_layers must be passed from config file if postnet is enabled"
          )

    self._output_projection_layer = tf.layers.Dense(
      self.num_audio_features, use_bias=True
    )
    self.target_projection_layer = tf.layers.Dense(
      1, use_bias=True
    )

    if self._mode == "train":
      dp_input_keep_prob = self.params.get('decoder_dp_input_keep_prob', 1.0)
      dp_output_keep_prob = self.params.get('decoder_dp_output_keep_prob', 1.0)
    else:
      dp_input_keep_prob = 1.0
      dp_output_keep_prob = 1.0


    residual_connections = self.params.get('decoder_use_skip_connections', False)
    wrap_to_multi_rnn = True

    cell_params = {}
    cell_params["num_units"] = self.params['decoder_cell_units']
    self._decoder_cells = create_rnn_cell(
      cell_type=self.params['decoder_cell_type'],
      cell_params=cell_params,
      num_layers=self.params['decoder_layers'],
      dp_input_keep_prob=dp_input_keep_prob,
      dp_output_keep_prob=dp_output_keep_prob,
      residual_connections=residual_connections,
      wrap_to_multi_rnn=wrap_to_multi_rnn,
    )


    if self.params['attention_type'] is not None:
      attention_mechanism = self._build_attention(
        encoder_outputs,
        enc_src_lengths,
        self.params.get("attention_bias", False)
      )

      attention_cell = self._decoder_cells
      if self.params["attention_rnn_enable"]:
        attention_rnn_units = self.params.get('attention_rnn_units', 1024)
        attention_rnn_layers = self.params.get('attention_rnn_layers', 1)
        cell_type = self.params.get('attention_rnn_cell_type', 'lstm')
        cell_params = {}
        cell_params["num_units"] = attention_rnn_units
        self._attention_cells = create_rnn_cell(
          cell_type=cell_type,
          cell_params=cell_params,
          num_layers=attention_rnn_layers,
          dp_input_keep_prob=dp_input_keep_prob,
          dp_output_keep_prob=dp_output_keep_prob,
          residual_connections=residual_connections,
          wrap_to_multi_rnn=wrap_to_multi_rnn,
        )
        attention_cell = self._attention_cells

      if self.params['attention_type'].startswith('gnmt'):
        raise ValueError("Attention type is currently not supported")
        # attention_cell = attention_cell.pop(0)
        # # attention_cell = tf.contrib.seq2seq.AttentionWrapper(
        # attention_cell = AttentionWrapper(
        #   attention_cell,
        #   attention_mechanism=attention_mechanism,
        #   attention_layer_size=None,
        #   output_attention=False,
        #   name="gnmt_attention")
        # attentive_cell = GNMTAttentionMultiCell(
        #   attention_cell, self._add_residual_wrapper(self._decoder_cells),
        #   use_new_attention=(self.params['attention_type'] == 'gnmt_v2'))
      elif self.params['attention_type'] == "luong":
        raise ValueError("Attention type is currently not supported")
        # attentive_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        # attentive_cell = AttentionWrapper(
        #   cell=attention_cell,
        #   attention_mechanism=attention_mechanism,
        #   alignment_history=True
        # )
      elif self.params['attention_type'] == "bahdanau":
        if self.params["attention_rnn_enable"]:
          output_attention = True
        else:
          output_attention = "both"
        attentive_cell = AttentionWrapper(
          cell=attention_cell,
          attention_mechanism=attention_mechanism,
          alignment_history=True,
          output_attention=output_attention
        )
      else:
        if self.params["attention_rnn_enable"]:
          output_attention = True
        else:
          output_attention = "both"
        attentive_cell = AttentionWrapper(
          cell=attention_cell,
          attention_mechanism=attention_mechanism,
          alignment_history=True,
          output_attention=output_attention
        )

      if not self.params["attention_rnn_enable"]:
        decoder_cell = attentive_cell
        initial_state = decoder_cell.zero_state(
            _batch_size, dtype=encoder_outputs.dtype,
          )
      else:
        decoder_cell = self._decoder_cells
        initial_state = self._decoder_cells.zero_state(
            _batch_size, dtype=encoder_outputs.dtype,
          )
    else:
      decoder_cell = self._decoder_cells
      initial_state = self._decoder_cells.zero_state(_batch_size, tf.float32)
      # if self.params['decoder_layers'] == 1:
      #   initial_state = enc_state[0]
      # else:
      #   initial_state = enc_state

    prenet_activation = self.params.get("prenet_activation", tf.nn.relu)
    mask_decoder_sequence = self.params.get("mask_decoder_sequence", True)
    if self._mode == "train":
      if self.params.get('anneal_sampling_prob', False):
        if "128" in self.model.get_data_layer().params['dataset_files'][0]:
          train_size = 128.
        else:
          train_size = 10480.
        curr_epoch = tf.div(tf.cast(tf.train.get_or_create_global_step(),tf.float32), tf.constant(train_size/_batch_size, tf.float32))
        curr_step = tf.floor(tf.div(curr_epoch,tf.constant(self.model.params["num_epochs"]/20.)))
        sampling_prob = tf.div(curr_step,tf.constant(20.))
      else:
        sampling_prob = self.params['scheduled_sampling_prob']
      helper = TacotronTrainingHelper(inputs=spec,
                                      sequence_length=spec_length,
                                      enable_prenet=enable_prenet,
                                      prenet_units=prenet_units,
                                      prenet_layers=prenet_layers,
                                      prenet_activation=prenet_activation,
                                      sampling_prob=sampling_prob,
                                      anneal_sampling_prob=self.params.get('anneal_sampling_prob', False),
                                      sampling_test=self.params.get("sampling_test",False),
                                      mask_decoder_sequence=mask_decoder_sequence)
                                      # context=mean_pool)
      # helper = TacotronHelper(inputs=tgt_inputs,
      #                         sequence_length=tgt_lengths,
      #                         enable_prenet=enable_prenet,
      #                         prenet_units=prenet_layers,
      #                         prenet_layers=prenet_layers,
      #                                 context=mean_pool)
      # helper = tf.contrib.seq2seq.TrainingHelper(
      #   inputs=tgt_inputs,
      #   sequence_length=tgt_lengths)
    elif self._mode == "eval" or self._mode == "infer":
      # embedding_fn = lambda ids: tf.cast(tf.nn.embedding_lookup(self._dec_emb_w, ids),
      #                         dtype=self.params['dtype'])
      inputs = tf.zeros((_batch_size, self.num_audio_features))
      helper = TacotronHelper(inputs=inputs,
                              # sequence_length=spec_length,
                              enable_prenet=enable_prenet,
                              prenet_units=prenet_units,
                              prenet_layers=prenet_layers,
                              prenet_activation=prenet_activation,
                              mask_decoder_sequence=mask_decoder_sequence)
                              # context=mean_pool)
      # helper = tf.contrib.seq2seq.TrainingHelper(
      #   inputs=encoder_outputs,
      #   sequence_length=enc_src_lengths)
    # elif self._mode == "infer":
    #   inputs = tf.zeros((_batch_size, self.num_audio_features))
    #   # tgt_lengths = None
    #   helper = TacotronHelper(inputs=inputs,
    #                           sequence_length=spec_length,
    #                           enable_prenet=enable_prenet,
    #                           prenet_units=prenet_units,
    #                           prenet_layers=prenet_layers,
    #                           prenet_activation=prenet_activation,
    #                           mask_decoder_sequence=mask_decoder_sequence)
    else:
      raise ValueError(
        "Unknown mode for decoder: {}".format(self._mode)
      )

    if not self.params["attention_rnn_enable"]:
      decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=decoder_cell,
        helper=helper,
        initial_state=initial_state,
        output_layer=self._output_projection_layer,
      )
    else:
      decoder = TacotronDecoder(
        decoder_cell=decoder_cell,
        attention_cell=attentive_cell,
        helper=helper,
        initial_decoder_state=initial_state,
        initial_attention_state=attentive_cell.zero_state(_batch_size, tf.float32),
        attention_type = self.params["attention_type"],
        spec_layer=self._output_projection_layer,
        target_layer=self.target_projection_layer,
        use_prenet_output = self.params.get("use_prenet_output", True)
      )

    time_major = self.params.get("time_major", False)
    use_swap_memory = self.params.get("use_swap_memory", False)
    if self._mode == 'train' or self._mode == 'eval':
      maximum_iterations = tf.reduce_max(spec_length)
    else:
      maximum_iterations = tf.reduce_max(enc_src_lengths) * 5
    # maximum_iterations = tf.reduce_max(spec_length)

    final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
    # final_outputs, final_state, final_sequence_lengths, final_inputs = dynamic_decode(
      decoder=decoder,
      # impute_finished=False if self._decoder_type == "beam_search" else True,
      impute_finished=True,
      maximum_iterations=maximum_iterations,
      swap_memory=use_swap_memory,
      output_time_major=time_major,
    )

    ## Add the post net ##
    if self.params.get('enable_postnet', True):
      enable_postnet_bn = self.params.get('enable_postnet_bn', True)
      use_bias = self.params.get('postnet_use_bias', True)
      dropout_keep_prob = self.params.get('postnet_keep_dropout_prob', 0.5)

      conv_layers = self.params['postnet_conv_layers']
      top_layer = final_outputs.rnn_output
      for idx_conv in range(len(conv_layers)):
        ch_out = conv_layers[idx_conv]['num_channels']
        kernel_size = conv_layers[idx_conv]['kernel_size']  # [time, freq]
        strides = conv_layers[idx_conv]['stride']
        padding = conv_layers[idx_conv]['padding']
        activation_fn = conv_layers[idx_conv]['activation_fn']

        if padding == "VALID":
          final_sequence_lengths = (final_sequence_lengths - kernel_size[0] + strides[0]) // strides[0]
        else:
          final_sequence_lengths = (final_sequence_lengths + strides[0] - 1) // strides[0]

        top_layer = conv1d_bn_actv(
          name="conv{}".format(idx_conv + 1),
          inputs=top_layer,
          filters=ch_out,
          kernel_size=kernel_size,
          activation_fn=activation_fn,
          strides=strides,
          padding=padding,
          regularizer=regularizer,
          training=training,
          use_bias=use_bias,
          data_format=data_format,
          enable_bn=enable_postnet_bn,
          bn_momentum=bn_momentum,
          bn_epsilon=bn_epsilon,
        )
        top_layer = tf.layers.dropout(top_layer, rate=1.-dropout_keep_prob, training=training)

    else:
      top_layer = tf.zeros([_batch_size, maximum_iterations, final_outputs.rnn_output.get_shape()[-1]])
    if self.params['attention_type'] is not None:
      if self.params['attention_rnn_enable']:
        alignments = tf.transpose(final_state[0].alignment_history.stack(), [1,0,2])
      else:
        alignments = tf.transpose(final_state.alignment_history.stack(), [1,0,2])
    else:
      alignments = tf.zeros([_batch_size,_batch_size,_batch_size])

    return {'decoder_output': final_outputs.rnn_output,
            'post_net_output': top_layer,
            'alignments': alignments,
            'final_sequence_lengths': final_sequence_lengths,
            'target_output': final_outputs.target_output}


# class BasicDecoder(Decoder):
#   """
#   Typical RNN decoder with attention mechanism
#   """
#   @staticmethod
#   def get_required_params():
#     return dict(Decoder.get_required_params(), **{
#       'attention_layer_size': int,
#       'attention_type': ['bahdanau', 'luong', 'gnmt', 'gnmt_v2', None],
#       'decoder_cell_units': int,
#       # 'decoder_proj_size': int,
#       'decoder_cell_type': ['lstm', 'gru', 'glstm', 'slstm'],
#       'decoder_layers': int,
#       'decoder_use_skip_connections': bool,
#       'num_audio_features': int,
#       # 'conv_layers': list,
#       'dropout_keep_prob': float

#     })

#   @staticmethod
#   def get_optional_params():
#     return dict(Decoder.get_optional_params(), **{
#       'bahdanau_normalize': bool,
#       'luong_scale': bool,
#       'decoder_dp_input_keep_prob': float,
#       'decoder_dp_output_keep_prob': float,
#       'time_major': bool,
#       # 'use_swap_memory': bool,
#       # 'proj_size': int,
#       # 'num_groups': int,
#       'PAD_SYMBOL': int,  # symbol id
#       'bn_momentum': float,
#       'bn_epsilon': float,
#       # 'data_format': ['channels_first', 'channels_last']
#     })

#   def __init__(self, params, model,
#                name='tacotron_2_decoder', mode='train'):
#     """
#     Initializes RNN decoder with embedding
#     :param params: dictionary with decoder parameters
#     Must define:
#       * batch_size - batch size
#       * GO_SYMBOL - GO symbol id, must be the same as used in data layer
#       * END_SYMBOL - END symbol id, must be the same as used in data layer
#       * tgt_vocab_size - vocabulary size of target
#       * tgt_emb_size - embedding to use
#       * decoder_cell_units - number of units in RNN
#       * decoder_cell_type - RNN type: lstm, gru, glstm, etc.
#       * decoder_dp_input_keep_prob -
#       * decoder_dp_output_keep_prob -
#       * decoder_use_skip_connections - use residual connections or not
#       * attention_type - bahdanau, luong, gnmt, gnmt_v2
#       * bahdanau_normalize - (optional)
#       * luong_scale - (optional)
#       * mode - train or infer
#       ... add any cell-specific parameters here as well
#     """
#     super(BasicDecoder, self).__init__(params, model, name, mode)
#     self.num_audio_features = self.params['num_audio_features']

#   def _build_attention(self,
#                        encoder_outputs,
#                        encoder_sequence_length):
#     """
#     Builds Attention part of the graph.
#     Currently supports "bahdanau" and "luong"
#     :param encoder_outputs:
#     :param encoder_sequence_length:
#     :return:
#     """
#     with tf.variable_scope("AttentionMechanism"):
#       attention_depth = self.params['attention_layer_size']
#       if self.params['attention_type'] == 'bahdanau':
#         if 'bahdanau_normalize' in self.params:
#           bah_normalize = self.params['bahdanau_normalize']
#         else:
#           bah_normalize = False
#         # attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
#         attention_mechanism = BahdanauAttention(
#           num_units=attention_depth,
#           memory=encoder_outputs,
#           normalize=bah_normalize,
#           memory_sequence_length=encoder_sequence_length,
#           probability_fn=tf.nn.softmax,
#           dtype=tf.get_variable_scope().dtype
#         )
#       elif self.params['attention_type'] == 'luong':
#         if 'luong_scale' in self.params:
#           luong_scale = self.params['luong_scale']
#         else:
#           luong_scale = False
#         # attention_mechanism = tf.contrib.seq2seq.LuongAttention(
#         attention_mechanism = LuongAttention(
#           num_units=attention_depth,
#           memory=encoder_outputs,
#           scale=luong_scale,
#           memory_sequence_length=encoder_sequence_length,
#           probability_fn=tf.nn.softmax,
#           dtype=tf.get_variable_scope().dtype
#         )
#       elif self.params['attention_type'] == 'gnmt' or \
#            self.params['attention_type'] == 'gnmt_v2':
#         # attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
#         attention_mechanism = BahdanauAttention(
#           num_units=attention_depth,
#           memory=encoder_outputs,
#           normalize=True,
#           memory_sequence_length=encoder_sequence_length,
#           probability_fn=tf.nn.softmax,
#           dtype=tf.get_variable_scope().dtype
#         )
#       else:
#         raise ValueError('Unknown Attention Type')
#       return attention_mechanism

#   @staticmethod
#   def _add_residual_wrapper(cells, start_ind=1):
#     for idx, cell in enumerate(cells):
#       if idx >= start_ind:
#         cells[idx] = tf.contrib.rnn.ResidualWrapper(
#           cell,
#           residual_fn=gnmt_residual_fn,
#         )
#     return cells

#   def _decode(self, input_dict):
#     """
#     Decodes representation into data
#     :param input_dict: Python dictionary with inputs to decoder
#     Must define:
#       * src_inputs - decoder input Tensor of shape [batch_size, time, dim]
#                      or [time, batch_size, dim]
#       * src_lengths - decoder input lengths Tensor of shape [batch_size]
#       * tgt_inputs - Only during training. labels Tensor of the
#                      shape [batch_size, time] or [time, batch_size]
#       * tgt_lengths - Only during training. labels lengths
#                       Tensor of the shape [batch_size]
#     :return: a Python dictionary with:
#       * final_outputs - tensor of shape [batch_size, time, dim]
#                         or [time, batch_size, dim]
#       * final_state - tensor with decoder final state
#       * final_sequence_lengths - tensor of shape [batch_size, time]
#                                  or [time, batch_size]
#     """
#     encoder_outputs = input_dict['encoder_output']['outputs']
#     enc_src_lengths = input_dict['encoder_output']['src_length']
#     enc_state = input_dict['encoder_output']['state']
#     tgt_inputs = input_dict['target_tensors'][0] if 'target_tensors' in \
#                                                     input_dict else None
#     tgt_lengths = input_dict['target_tensors'][1] if 'target_tensors' in \
#                                                     input_dict else None
#     # print(tgt_inputs)
#     # print(tgt_lengths)
#     # input()
#     _batch_size = encoder_outputs.get_shape().as_list()[0]

#     training = (self._mode == "train")
#     dropout_keep_prob = self.params['dropout_keep_prob'] if training else 1.0
#     regularizer = self.params.get('regularizer', None)
#     bn_momentum = self.params.get('bn_momentum', 0.99)
#     bn_epsilon = self.params.get('bn_epsilon', 1e-3)
#     # data_format = self.params.get('data_format', 'channels_last')
#     # print(tgt_lengths)

#     # self._dec_emb_w = tf.get_variable(
#     #   name='DecoderEmbeddingMatrix',
#     #   shape=[self._tgt_vocab_size, self._tgt_emb_size],
#     #   dtype=tf.float32,
#     # )

#     self._output_projection_layer = tf.layers.Dense(
#       self.num_audio_features, use_bias=False
#     )

#     cell_params = copy.deepcopy(self.params)
#     cell_params["num_units"] = self.params['decoder_cell_units']
#     # cell_params["proj_size"] = self.params['decoder_proj_size']

#     if self._mode == "train":
#       dp_input_keep_prob = self.params.get('decoder_dp_input_keep_prob', 1.0)
#       dp_output_keep_prob = self.params.get('decoder_dp_output_keep_prob', 1.0)
#     else:
#       dp_input_keep_prob = 1.0
#       dp_output_keep_prob = 1.0

#     if type(self.params['attention_type']) == str and self.params['attention_type'].startswith('gnmt'):
#       residual_connections = False
#       wrap_to_multi_rnn = False
#     else:
#       residual_connections = self.params['decoder_use_skip_connections']
#       wrap_to_multi_rnn = True

#     self._decoder_cells = create_rnn_cell(
#       cell_type=self.params['decoder_cell_type'],
#       cell_params=cell_params,
#       num_layers=self.params['decoder_layers'],
#       dp_input_keep_prob=dp_input_keep_prob,
#       dp_output_keep_prob=dp_output_keep_prob,
#       residual_connections=residual_connections,
#       wrap_to_multi_rnn=wrap_to_multi_rnn,
#     )

#     # print(encoder_outputs.shape)
#     if self.params['attention_type'] is not None:
#       attention_mechanism = self._build_attention(
#         encoder_outputs,
#         enc_src_lengths,
#       )
#       if self.params['attention_type'].startswith('gnmt'):
#         attention_cell = self._decoder_cells.pop(0)
#         # attention_cell = tf.contrib.seq2seq.AttentionWrapper(
#         attention_cell = AttentionWrapper(
#           attention_cell,
#           attention_mechanism=attention_mechanism,
#           attention_layer_size=None,
#           output_attention=False,
#           name="gnmt_attention")
#         attentive_decoder_cell = GNMTAttentionMultiCell(
#           attention_cell, self._add_residual_wrapper(self._decoder_cells),
#           use_new_attention=(self.params['attention_type'] == 'gnmt_v2'))
#       else:
#         # attentive_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
#         attentive_decoder_cell = AttentionWrapper(
#           cell=self._decoder_cells,
#           attention_mechanism=attention_mechanism,
#           alignment_history=True
#         )
#       decoder_cell = attentive_decoder_cell
#     else:
#       decoder_cell = self._decoder_cells

#     # print(enc_state)
#     # print(enc_state[0])
#     # input()
#     # print(decoder_cell.zero_state(batch_size=_batch_size, dtype=encoder_outputs.dtype))
#     # print(enc_state)
#     # input()
#     if self.params['decoder_layers'] == 1:
#       enc_state = enc_state[0]
#     #   enc_state = (enc_state[i] for i in range(self.params['decoder_layers']))
#     # else:
#     # enc_state = [enc_state[0], enc_state[1]]
#     # input()
#     # enc_state = tf.concat([enc_state]*2, 0)
#     # enc_state = (LSTMStateTuple(enc_state[0],enc_state[1]),LSTMStateTuple(enc_state[2],enc_state[3]))
#     # input()

#     if self._mode == "train":
#       # input_vectors = tf.cast(tf.nn.embedding_lookup(self._dec_emb_w, tgt_inputs),
#       #                         dtype=self.params['dtype'])
#       # helper = TacotronHelper(inputs=tgt_inputs,
#       #                           sequence_length=tgt_lengths)
#       helper = TacotronTrainingHelper(inputs=tgt_inputs,
#                                 sequence_length=tgt_lengths)
#       # helper = ScheduledSamplingHelper(inputs=tgt_inputs,
#       #                           sequence_length=tgt_lengths,
#       #                           sampling_probability = 0.5,
#       #                           seed = 0)
#       # helper = tf.contrib.seq2seq.TrainingHelper(
#       #   inputs=tgt_inputs,
#       #   sequence_length=tgt_lengths)
#       decoder = tf.contrib.seq2seq.BasicDecoder(
#         cell=decoder_cell,
#         helper=helper,
#         output_layer=self._output_projection_layer,
#         initial_state=enc_state
#       )
#     elif self._mode == "infer" or self._mode == "eval":
#       # embedding_fn = lambda ids: tf.cast(tf.nn.embedding_lookup(self._dec_emb_w, ids),
#       #                         dtype=self.params['dtype'])
#       helper = TacotronHelper(inputs=tgt_inputs,
#                               sequence_length=tgt_lengths)
#       # helper = tf.contrib.seq2seq.TrainingHelper(
#       #   inputs=encoder_outputs,
#       #   sequence_length=enc_src_lengths)
#       decoder = tf.contrib.seq2seq.BasicDecoder(
#         cell=decoder_cell,
#         helper=helper,
#         initial_state=enc_state,
#         output_layer=self._output_projection_layer,
#       )
#     else:
#       raise ValueError(
#         "Unknown mode for decoder: {}".format(self._mode)
#       )

#     time_major = self.params.get("time_major", False)
#     use_swap_memory = self.params.get("use_swap_memory", False)
#     if self._mode == 'train' or self._mode == 'eval':
#       maximum_iterations = tf.reduce_max(tgt_lengths)
#     else:
#       maximum_iterations = tf.reduce_max(enc_src_lengths) * 2

#     final_outputs, final_state, final_sequence_lengths, final_inputs = tf.contrib.seq2seq.dynamic_decode(
#       decoder=decoder,
#       # impute_finished=False if self._decoder_type == "beam_search" else True,
#       impute_finished=True,
#       maximum_iterations=maximum_iterations,
#       swap_memory=use_swap_memory,
#       output_time_major=time_major,
#     )

#     # ## Add the post net ##
#     # conv_layers = self.params['conv_layers']
#     # top_layer = final_outputs.rnn_output
#     # for idx_conv in range(len(conv_layers)):
#     #   ch_out = conv_layers[idx_conv]['num_channels']
#     #   kernel_size = conv_layers[idx_conv]['kernel_size']  # [time, freq]
#     #   strides = conv_layers[idx_conv]['stride']
#     #   padding = conv_layers[idx_conv]['padding']
#     #   activation_fn = conv_layers[idx_conv]['activation_fn']

#     #   if padding == "VALID":
#     #     final_sequence_lengths = (final_sequence_lengths - kernel_size[0] + strides[0]) // strides[0]
#     #   else:
#     #     final_sequence_lengths = (final_sequence_lengths + strides[0] - 1) // strides[0]

#     #   top_layer = conv1d_bn_actv(
#     #     name="conv{}".format(idx_conv + 1),
#     #     inputs=top_layer,
#     #     filters=ch_out,
#     #     kernel_size=kernel_size,
#     #     activation_fn=activation_fn,
#     #     strides=strides,
#     #     padding=padding,
#     #     regularizer=regularizer,
#     #     training=training,
#     #     data_format=data_format,
#     #     bn_momentum=bn_momentum,
#     #     bn_epsilon=bn_epsilon,
#     #   )

#     if self.params['attention_type'] is not None:
#       alignments = tf.transpose(final_state.alignment_history.stack(), [1,0,2])
#     else:
#       alignments = tf.zeros([_batch_size,_batch_size,_batch_size])
#     return {'decoder_output': final_outputs.rnn_output,
#             # 'post_net_output': top_layer,
#             'alignments': alignments,
#             'final_sequence_lengths': final_sequence_lengths,
#             'final_inputs': final_inputs}
