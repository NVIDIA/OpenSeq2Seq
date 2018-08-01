# Copyright (c) 2018 NVIDIA Corporation
"""
RNN-based encoders
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from open_seq2seq.parts.rnns.utils import single_cell
from .encoder import Encoder
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

class AWDLSTMEncoder(Encoder):
  """
  Bi-directional RNN-based encoder with embeddings.
  Similar to the one used in AWD-LSTM (Merity et al.)
  """
  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
      'vocab_size': int,
      'emb_size': int,
      'encoder_layers': int,
      'encoder_use_skip_connections': bool,
      'core_cell': None,
      'core_cell_params': dict,
      'last_cell_params': dict,
      'output_dim': int,
      'end_token': int,
      "batch_size": int,
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
      'encoder_dp_input_keep_prob': float,
      'encoder_dp_output_keep_prob': float,
      'time_major': bool,
      'use_swap_memory': bool,
      'proj_size': int,
      'num_groups': int,
      'num_tokens_gen': int,
      'fc_use_bias': bool,
      'seed_tokens': list,
      'sampling_prob': float,
      'schedule_learning': bool,
      'weight_tied': bool,
    })

  def __init__(self, params, model,
               name="rnn_encoder_awd", mode='train'):
    """
    Initializes bi-directional encoder with embeddings
    :param params: dictionary with encoder parameters
    Must define:
      * vocab_size - data vocabulary size
      * emb_size - size of embedding to use
      * encoder_cell_units - number of units in RNN cell
      * encoder_cell_type - cell type: lstm, gru, etc.
      * encoder_layers - number of layers
      * encoder_dp_input_keep_prob -
      * encoder_dp_output_keep_prob -
      * encoder_use_skip_connections - true/false
      * time_major (optional)
      * use_swap_memory (optional)
      * mode - train or infer
      ... add any cell-specific parameters here as well
    :param encoder_params:
    """
    super(AWDLSTMEncoder, self).__init__(
      params, model, name=name, mode=mode,
    )
    self._vocab_size = self.params['vocab_size']
    self._emb_size = self.params['emb_size']
    self._sampling_prob = self.params.get('sampling_prob', 0.0)
    self._schedule_learning = self.params.get('schedule_learning', False)
    self._weight_tied = self.params.get('weight_tied', False)
    if mode == 'infer':
      self.num_tokens_gen = self.params.get('num_tokens_gen', 1)
      self._batch_size = len(self.params['seed_tokens'])
    else:
      self.num_tokens_gen = 1
      self._batch_size = self.params['batch_size']

  def _encode(self, input_dict):
    """
    Encodes data into representation
    :param input_dict: a Python dictionary.
    Must define:
      * src_inputs - a Tensor of shape [batch_size, time] or [time, batch_size]
                    (depending on time_major param)
      * src_lengths - a Tensor of shape [batch_size]
    :return: a Python dictionary with:
      * encoder_outputs - a Tensor of shape
                          [batch_size, time, representation_dim]
      or [time, batch_size, representation_dim]
      * encoder_state - a Tensor of shape [batch_size, dim]
      * src_lengths - (copy ref from input) a Tensor of shape [batch_size]
    """
    time_major = self.params.get("time_major", False)
    use_swap_memory = self.params.get("use_swap_memory", False)

    regularizer = self.params.get('regularizer', None)
    fc_use_bias = self.params.get('fc_use_bias', True)

    self._output_layer = tf.layers.Dense(
      self._vocab_size, 
      kernel_regularizer=regularizer,
      use_bias=fc_use_bias,
    )

    if self._weight_tied:
      fake_input = tf.zeros(shape=(1, self._emb_size))
      fake_output = self._output_layer.apply(fake_input)
      with tf.variable_scope("dense", reuse=True):
        self._enc_emb_w = tf.transpose(tf.get_variable("kernel"))
        
    else:
      self._enc_emb_w = tf.get_variable(
        name="EncoderEmbeddingMatrix",
        shape=[self._vocab_size, self._emb_size],
        dtype=tf.float32
      )
      
    if self._mode == "train":
      dp_input_keep_prob = self.params['encoder_dp_input_keep_prob']
      dp_output_keep_prob = self.params['encoder_dp_output_keep_prob']
    else:
      dp_input_keep_prob = 1.0
      dp_output_keep_prob = 1.0

    if self._weight_tied:
      last_cell_params = self.params['last_cell_params']
    else:
      last_cell_params = self.params['core_cell_params']

    fwd_cells = [
      single_cell(cell_class=self.params['core_cell'],
                  cell_params=self.params['core_cell_params'],
                  dp_input_keep_prob=dp_input_keep_prob,
                  dp_output_keep_prob=dp_output_keep_prob,
                  residual_connections=self.params['encoder_use_skip_connections']
                  ) for _ in range(self.params['encoder_layers'] - 1)]

    fwd_cells.append(
      single_cell(cell_class=self.params['core_cell'],
                  cell_params=last_cell_params,
                  dp_input_keep_prob=dp_input_keep_prob,
                  dp_output_keep_prob=dp_output_keep_prob,
                  residual_connections=self.params['encoder_use_skip_connections']
                  )
      )

    self._encoder_cell_fw = tf.contrib.rnn.MultiRNNCell(fwd_cells)

    time_major = self.params.get("time_major", False)
    use_swap_memory = self.params.get("use_swap_memory", False)

    source_sequence = input_dict['source_tensors'][0]
    source_length = input_dict['source_tensors'][1]

    if self._mode == 'train' or self._mode == 'eval':
      input_vectors = tf.cast(tf.nn.embedding_lookup(
        self.enc_emb_w,
        source_sequence,
      ), self.params['dtype'])

      if self._schedule_learning:
        embedding_fn = lambda ids: tf.cast(tf.nn.embedding_lookup(
                                            self.enc_emb_w,
                                            ids,
                                          ), self.params['dtype'])

        helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
          inputs=source_sequence,
          sequence_length=source_length,
          time_major=time_major,
          embedding=embedding_fn,
          sampling_probability=tf.constant(self._sampling_prob))
      else:
        helper = tf.contrib.seq2seq.TrainingHelper(
          inputs=input_vectors,
          sequence_length=source_length,
          time_major=time_major)
      
      decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=self._encoder_cell_fw,
        helper=helper,
        output_layer=self._output_layer,
        initial_state=self._encoder_cell_fw.zero_state(
          self._batch_size, dtype=tf.float32,
        ),
      )
      maximum_iterations = tf.reduce_max(source_length)

    else:
      embedding_fn = lambda ids: tf.cast(tf.nn.embedding_lookup(
                                          self.enc_emb_w,
                                          ids,
                                        ), self.params['dtype'])

      helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embedding=embedding_fn,#self._dec_emb_w,
        start_tokens = tf.constant(self.params['seed_tokens']),
        end_token=self.params['end_token'])
      
      decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=self._encoder_cell_fw,
        helper=helper,
        initial_state=self._encoder_cell_fw.zero_state(
          batch_size=self._batch_size, dtype=tf.float32,
        ),
        output_layer=self._output_layer,
      )
      maximum_iterations = tf.constant(200)

    final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
      decoder=decoder,
      impute_finished=False,
      maximum_iterations=maximum_iterations,
      swap_memory=use_swap_memory,
      output_time_major=time_major,
    )

    output_dict = {'logits': final_outputs.rnn_output,
          'outputs': [tf.argmax(final_outputs.rnn_output, axis=-1)],
          'final_state': final_state,
          'final_sequence_lengths': final_sequence_lengths}

    # for v in tf.trainable_variables():
    #   print(v.name)
    #   print(v.shape)
    

    return output_dict

# class AWDLSTMNoGenEncoder(Encoder):
#   """
#   Bi-directional RNN-based encoder with embeddings.
#   Similar to the one used in AWD-LSTM (Merity et al.)
#   """
#   @staticmethod
#   def get_required_params():
#     return dict(Encoder.get_required_params(), **{
#       'vocab_size': int,
#       'emb_size': int,
#       'encoder_layers': int,
#       'encoder_use_skip_connections': bool,
#       'core_cell': None,
#       'core_cell_params': dict,
#       'last_cell_params': dict,
#       'output_dim': int,
#       'end_token': int,
#       "batch_size": int,
#     })

#   @staticmethod
#   def get_optional_params():
#     return dict(Encoder.get_optional_params(), **{
#       'encoder_dp_input_keep_prob': float,
#       'encoder_dp_output_keep_prob': float,
#       'time_major': bool,
#       'use_swap_memory': bool,
#       'proj_size': int,
#       'num_groups': int,
#       'num_tokens_gen': int,
#       'fc_use_bias': bool,
#       'seed_token': int,
#     })

#   def __init__(self, params, model,
#                name="rnn_encoder_awd", mode='train'):
#     """
#     Initializes bi-directional encoder with embeddings
#     :param params: dictionary with encoder parameters
#     Must define:
#       * vocab_size - data vocabulary size
#       * emb_size - size of embedding to use
#       * encoder_cell_units - number of units in RNN cell
#       * encoder_cell_type - cell type: lstm, gru, etc.
#       * encoder_layers - number of layers
#       * encoder_dp_input_keep_prob -
#       * encoder_dp_output_keep_prob -
#       * encoder_use_skip_connections - true/false
#       * time_major (optional)
#       * use_swap_memory (optional)
#       * mode - train or infer
#       ... add any cell-specific parameters here as well
#     :param encoder_params:
#     """
#     super(AWDLSTMEncoder, self).__init__(
#       params, model, name=name, mode=mode,
#     )
#     self._vocab_size = self.params['vocab_size']
#     self._emb_size = self.params['emb_size']
#     if mode == 'infer':
#       self.num_tokens_gen = self.params.get('num_tokens_gen', 1)
#       self._batch_size = 1
#     else:
#       self.num_tokens_gen = 1
#       self._batch_size = self.params['batch_size']




#   # def _get_logits(self, encoder_outputs):
#   #   regularizer = self.params.get('regularizer', None)

#   #   # activation is linear by default
#   #   logits = tf.layers.dense(
#   #     inputs=encoder_outputs,
#   #     units=self.params['output_dim'],
#   #     kernel_regularizer=regularizer,
#   #     name='fully_connected',
#   #   )
#   #   return {'logits': logits, 'outputs': [logits]}

#   def _encode(self, input_dict):
#     """
#     Encodes data into representation
#     :param input_dict: a Python dictionary.
#     Must define:
#       * src_inputs - a Tensor of shape [batch_size, time] or [time, batch_size]
#                     (depending on time_major param)
#       * src_lengths - a Tensor of shape [batch_size]
#     :return: a Python dictionary with:
#       * encoder_outputs - a Tensor of shape
#                           [batch_size, time, representation_dim]
#       or [time, batch_size, representation_dim]
#       * encoder_state - a Tensor of shape [batch_size, dim]
#       * src_lengths - (copy ref from input) a Tensor of shape [batch_size]
#     """
#     time_major = self.params.get("time_major", False)
#     use_swap_memory = self.params.get("use_swap_memory", False)

#     self._enc_emb_w = tf.get_variable(
#       name="EncoderEmbeddingMatrix",
#       shape=[self._vocab_size, self._emb_size],
#       dtype=tf.float32
#     )

#     if self._mode == "train":
#       dp_input_keep_prob = self.params['encoder_dp_input_keep_prob']
#       dp_output_keep_prob = self.params['encoder_dp_output_keep_prob']
#     else:
#       dp_input_keep_prob = 1.0
#       dp_output_keep_prob = 1.0

#     fwd_cells = [
#       single_cell(cell_class=self.params['core_cell'],
#                   cell_params=self.params['core_cell_params'],
#                   dp_input_keep_prob=dp_input_keep_prob,
#                   dp_output_keep_prob=dp_output_keep_prob,
#                   residual_connections=self.params['encoder_use_skip_connections']
#                   ) for _ in range(self.params['encoder_layers'] - 1)]

#     fwd_cells.append(
#       single_cell(cell_class=self.params['core_cell'],
#                   cell_params=self.params['last_cell_params'],
#                   dp_input_keep_prob=dp_input_keep_prob,
#                   dp_output_keep_prob=dp_output_keep_prob,
#                   residual_connections=self.params['encoder_use_skip_connections']
#                   )
#       )

#     self._encoder_cell_fw = tf.contrib.rnn.MultiRNNCell(fwd_cells)

#     time_major = self.params.get("time_major", False)
#     use_swap_memory = self.params.get("use_swap_memory", False)

#     source_sequence = input_dict['source_tensors'][0]
#     source_length = input_dict['source_tensors'][1]

#     regularizer = self.params.get('regularizer', None)
#     fc_use_bias = self.params.get('fc_use_bias', True)

#     # fc_weights = tf.get_variable('fc_weights', 
#     #   shape=[self.params['last_cell_params']['num_units'], self._vocab_size],
#     #   initializer=tf.random_uniform_initializer)

#     # fc_biases = tf.get_variable('fc_biases',
#     #   shape=[self._vocab_size],
#     #   initializer=tf.random_uniform_initializer)

#     self._output_layer = tf.layers.Dense(
#       self._vocab_size, 
#       kernel_regularizer=regularizer,
#       use_bias=fc_use_bias,
#     )


#     if self._mode == 'train' or self._mode == 'eval':
#       embedded_inputs = tf.cast(tf.nn.embedding_lookup(
#         self.enc_emb_w,
#         source_sequence,
#       ), self.params['dtype'])

#       encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
#         cell=self._encoder_cell_fw,
#         inputs=embedded_inputs,
#         sequence_length=source_length,
#         time_major=time_major,
#         swap_memory=use_swap_memory,
#         dtype=embedded_inputs.dtype,
#       )

#       logits = self._output_layer.apply(encoder_outputs)
#       print(encoder_outputs.dtype)

#       return {'logits': logits, 'outputs': [logits]}

#     else:
#       print('seed_token', (self.params['seed_token']))
#       embedding_fn = lambda ids: tf.cast(tf.nn.embedding_lookup(
#                                           self.enc_emb_w,
#                                           ids,
#                                         ), self.params['dtype'])

#       helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
#         embedding=embedding_fn,#self._dec_emb_w,
#         start_tokens=tf.fill([self._batch_size], self.params['seed_token']),
#         end_token=self.params['end_token'])
#       decoder = tf.contrib.seq2seq.BasicDecoder(
#         cell=self._encoder_cell_fw,
#         helper=helper,
#         initial_state=self._encoder_cell_fw.zero_state(
#           batch_size=self._batch_size, dtype=tf.float32,
#         ),
#         output_layer=self._output_layer,
#       )
#       maximum_iterations = 200

#       final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
#         decoder=decoder,
#         # impute_finished=False if self._decoder_type == "beam_search" else True,
#         impute_finished=True,
#         maximum_iterations=maximum_iterations,
#         swap_memory=use_swap_memory,
#         output_time_major=time_major,
#       )

#       return {'logits': final_outputs.rnn_output if not time_major else
#             tf.transpose(final_outputs.rnn_output, perm=[1, 0, 2]),
#             'outputs': [tf.argmax(final_outputs.rnn_output, axis=-1)],
#             'final_state': final_state,
#             'final_sequence_lengths': final_sequence_lengths}

    
    # return {'outputs': encoder_outputs,
    #         'state': encoder_state,
    #         'src_lengths': source_length,
    #         'encoder_input': source_sequence}

  @property
  def vocab_size(self):
    return self._vocab_size

  @property
  def emb_size(self):
    return self._emb_size

  @property
  def enc_emb_w(self):
    return self._enc_emb_w

class OldAWDLSTMEncoder(Encoder):
  """
  Bi-directional RNN-based encoder with embeddings.
  Similar to the one used in AWD-LSTM (Merity et al.)
  """
  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
      'vocab_size': int,
      'emb_size': int,
      'encoder_layers': int,
      'encoder_use_skip_connections': bool,
      'core_cell': None,
      'core_cell_params': dict,
      'last_cell_params': dict,
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
      'encoder_dp_input_keep_prob': float,
      'encoder_dp_output_keep_prob': float,
      'time_major': bool,
      'use_swap_memory': bool,
      'proj_size': int,
      'num_groups': int,
    })

  def __init__(self, params, model,
               name="rnn_encoder_awd", mode='train'):
    """
    Initializes bi-directional encoder with embeddings
    :param params: dictionary with encoder parameters
    Must define:
      * vocab_size - data vocabulary size
      * emb_size - size of embedding to use
      * encoder_cell_units - number of units in RNN cell
      * encoder_cell_type - cell type: lstm, gru, etc.
      * encoder_layers - number of layers
      * encoder_dp_input_keep_prob -
      * encoder_dp_output_keep_prob -
      * encoder_use_skip_connections - true/false
      * time_major (optional)
      * use_swap_memory (optional)
      * mode - train or infer
      ... add any cell-specific parameters here as well
    :param encoder_params:
    """
    super(OldAWDLSTMEncoder, self).__init__(
      params, model, name=name, mode=mode,
    )

    self._vocab_size = self.params['vocab_size']
    self._emb_size = self.params['emb_size']

  def _encode(self, input_dict):
    """
    Encodes data into representation
    :param input_dict: a Python dictionary.
    Must define:
      * src_inputs - a Tensor of shape [batch_size, time] or [time, batch_size]
                    (depending on time_major param)
      * src_lengths - a Tensor of shape [batch_size]
    :return: a Python dictionary with:
      * encoder_outputs - a Tensor of shape
                          [batch_size, time, representation_dim]
      or [time, batch_size, representation_dim]
      * encoder_state - a Tensor of shape [batch_size, dim]
      * src_lengths - (copy ref from input) a Tensor of shape [batch_size]
    """
    source_sequence = input_dict['source_tensors'][0]
    source_length = input_dict['source_tensors'][1]
    time_major = self.params.get("time_major", False)
    use_swap_memory = self.params.get("use_swap_memory", False)

    self._enc_emb_w = tf.get_variable(
      name="EncoderEmbeddingMatrix",
      shape=[self._vocab_size, self._emb_size],
      dtype=tf.float32
    )

    if self._mode == "train":
      dp_input_keep_prob = self.params['encoder_dp_input_keep_prob']
      dp_output_keep_prob = self.params['encoder_dp_output_keep_prob']
    else:
      dp_input_keep_prob = 1.0
      dp_output_keep_prob = 1.0

    fwd_cells = [
      single_cell(cell_class=self.params['core_cell'],
                  cell_params=self.params.get('core_cell_params', {}),
                  dp_input_keep_prob=dp_input_keep_prob,
                  dp_output_keep_prob=dp_output_keep_prob,
                  residual_connections=self.params['encoder_use_skip_connections']
                  ) for _ in range(self.params['encoder_layers'] - 1)]

    fwd_cells.append(
      single_cell(cell_class=self.params['core_cell'],
                  cell_params=self.params.get('last_cell_params', {}),
                  dp_input_keep_prob=dp_input_keep_prob,
                  dp_output_keep_prob=dp_output_keep_prob,
                  residual_connections=self.params['encoder_use_skip_connections']
                  )
      )

    self._encoder_cell_fw = tf.contrib.rnn.MultiRNNCell(fwd_cells)

    time_major = self.params.get("time_major", False)
    use_swap_memory = self.params.get("use_swap_memory", False)

    embedded_inputs = tf.cast(tf.nn.embedding_lookup(
      self.enc_emb_w,
      source_sequence,
    ), self.params['dtype'])

    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
      cell=self._encoder_cell_fw,
      inputs=embedded_inputs,
      sequence_length=source_length,
      time_major=time_major,
      swap_memory=use_swap_memory,
      dtype=embedded_inputs.dtype,
    )
    return {'outputs': encoder_outputs,
            'state': encoder_state,
            'src_lengths': source_length,
            'encoder_input': source_sequence}

  @property
  def vocab_size(self):
    return self._vocab_size

  @property
  def emb_size(self):
    return self._emb_size

  @property
  def enc_emb_w(self):
    return self._enc_emb_w