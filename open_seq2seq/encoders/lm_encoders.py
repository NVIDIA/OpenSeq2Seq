# Copyright (c) 2018 NVIDIA Corporation
"""
RNN-based encoders
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import copy, inspect
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from open_seq2seq.optimizers.mp_wrapper import mp_regularizer_wrapper
from open_seq2seq.parts.rnns.utils import single_cell
from .encoder import Encoder
# from open_seq2seq.parts.rnns.weight_drop import WeightDropLayerNormBasicLSTMCell


class LMEncoder(Encoder):
  """
  RNN-based encoder with embeddings for language modeling
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
      'end_token': int,
      "batch_size": int,
      "use_cudnn_rnn": bool,
      "cudnn_rnn_type": None
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
      'encoder_dp_input_keep_prob': float,
      'encoder_dp_output_keep_prob': float,
      "encoder_last_input_keep_prob": float,
      "encoder_last_output_keep_prob": float,
      'encoder_emb_keep_prob': float,
      'variational_recurrent': bool,
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
      'awd_initializer': bool,
      "recurrent_keep_prob": float,
      "input_weight_keep_prob": float,
      "recurrent_weight_keep_prob": float,
      "weight_variational": bool,
      "dropout_seed": int,
      "num_sampled": int,
      "fc_dim": int,
      "use_cell_state": bool,
    })

  def __init__(self, params, model,
               name="rnn_encoder_awd", mode='train'):
    """
    Initializes bi-directional encoder with embeddings
    :param params: dictionary with encoder parameters

    Many of the techniques in this implementation is taken from the paper
    "Regularizing and Optimizing LSTM Language Models" (Merity et al., 2017)
    https://arxiv.org/pdf/1708.02182.pdf

    Must define:
      * vocab_size - data vocabulary size
      * emb_size - size of embedding to use
      * encoder_cell_units - number of units in RNN cell
      * encoder_cell_type - cell type: lstm, gru, etc.
      * encoder_layers - number of layers
      * encoder_use_skip_connections - true/false
      * time_major (optional)
      * use_swap_memory (optional)
      * mode - train or infer
      * input_weight_keep_prob: keep probability for dropout of W 
                                (kernel used to multiply with the input tensor)
      * recurrent_weight_keep_prob: keep probability for dropout of U
                                  (kernel used to multiply with last hidden state tensor)
      * recurrent_keep_prob: keep probability for dropout
                            when applying tanh for the input transform step
      * weight_variational: whether to keep the same weight dropout mask
                            at every timestep. This feature is not yet implemented.
      * emb_keep_prob: keep probability for dropout of the embedding matrix
      * encoder_dp_input_keep_prob: keep probability for dropout on input of a LSTM cell
                                    in the layer which is not the last layer
      * encoder_dp_output_keep_prob: keep probability for dropout on output of a LSTM cell
                                    in the layer which is not the last layer
      * encoder_last_input_keep_prob: like ``encoder_dp_input_keep_prob`` but for the 
                                      cell in the last layer
      * encoder_dp_output_keep_prob: like ``encoder_dp_output_keep_prob`` but for the 
                                      cell in the last layer
      * weight_tied: whether to tie the embedding matrix to the last linear layer.
                     can only do so if the dimension of the last output layer is
                     the same as the vocabulary size
      * use_cell_state: if set to True, concat the last hidden state and 
                        the last cell state to input into the last output layer.
                        This only works for the text classification task, not the
                        language modeling phase.
      For different ways to do dropout for LSTM cells, please read this article:
      https://medium.com/@bingobee01/a-review-of-dropout-as-applied-to-rnns-72e79ecd5b7b

    :param encoder_params:
    """
    super(LMEncoder, self).__init__(
      params, model, name=name, mode=mode,
    )
    self._vocab_size = self.params['vocab_size']
    self._emb_size = self.params['emb_size']
    self._sampling_prob = self.params.get('sampling_prob', 0.0)
    self._schedule_learning = self.params.get('schedule_learning', False)
    self._weight_tied = self.params.get('weight_tied', False)
    self.params['encoder_last_input_keep_prob'] = self.params.get('encoder_last_input_keep_prob', 1.0)
    self.params['encoder_last_output_keep_prob'] = self.params.get('encoder_last_output_keep_prob', 1.0)
    self.params['encoder_emb_keep_prob'] = self.params.get('encoder_emb_keep_prob', 1.0)
    self.params['variational_recurrent'] = self.params.get('variational_recurrent', False)
    self.params['awd_initializer'] = self.params.get('awd_initializer', False)
    self.params['recurrent_keep_prob'] = self.params.get('recurrent_keep_prob', 1.0)
    self.params['input_weight_keep_prob'] = self.params.get('input_weight_keep_prob', 1.0)
    self.params['recurrent_weight_keep_prob'] = self.params.get('recurrent_weight_keep_prob', 1.0)
    self.params['weight_variational'] = self.params.get('weight_variational', False)
    self.params['dropout_seed'] = self.params.get('dropout_seed', 1822)
    self._fc_dim = self.params.get('fc_dim', self._vocab_size)
    self._num_sampled = self.params.get('num_sampled', self._fc_dim) # if num_sampled not defined, take full softmax
    self._lm_phase = self._fc_dim == self._vocab_size
    self._num_tokens_gen = self.params.get('num_tokens_gen', 200)
    self._batch_size = self.params['batch_size']
    
    if mode == 'infer' and self._lm_phase:
      self._batch_size = len(self.params['seed_tokens'])
    self._use_cell_state = self.params.get('use_cell_state', False)

  def encode(self, input_dict):
    """Wrapper around :meth:`self._encode() <_encode>` method.
    Here name, initializer and dtype are set in the variable scope and then
    :meth:`self._encode() <_encode>` method is called.

    Args:
      input_dict (dict): see :meth:`self._encode() <_encode>` docs.

    Returns:
      see :meth:`self._encode() <_encode>` docs.
    """

    if not self._compiled:
      if 'regularizer' not in self._params:
        if self._model and 'regularizer' in self._model.params:
          self._params['regularizer'] = copy.deepcopy(
              self._model.params['regularizer']
          )
          self._params['regularizer_params'] = copy.deepcopy(
              self._model.params['regularizer_params']
          )

      if 'regularizer' in self._params:
        init_dict = self._params.get('regularizer_params', {})
        self._params['regularizer'] = self._params['regularizer'](**init_dict)
        if self._params['dtype'] == 'mixed':
          self._params['regularizer'] = mp_regularizer_wrapper(
              self._params['regularizer'],
          )

      if self._params['dtype'] == 'mixed':
        self._params['dtype'] = tf.float16

    self._compiled = True

    with tf.variable_scope(self._name, dtype=self.params['dtype']):
      return self._encode(self._cast_types(input_dict))

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

    use_cudnn_rnn = self.params.get("use_cudnn_rnn", False)
    cudnn_rnn_type = self.params.get("cudnn_rnn_type", None)

    if 'initializer' in self.params:
      init_dict = self.params.get('initializer_params', {})
      initializer = self.params['initializer'](**init_dict)
    else:
      initializer = None

    if self._mode == "train":
      dp_input_keep_prob = self.params['encoder_dp_input_keep_prob']
      dp_output_keep_prob = self.params['encoder_dp_output_keep_prob']
      last_input_keep_prob = self.params['encoder_last_input_keep_prob']
      last_output_keep_prob = self.params['encoder_last_output_keep_prob']
      emb_keep_prob = self.params['encoder_emb_keep_prob']
      recurrent_keep_prob = self.params['recurrent_keep_prob']
      input_weight_keep_prob = self.params['input_weight_keep_prob']
      recurrent_weight_keep_prob = self.params['recurrent_weight_keep_prob']

    else:
      dp_input_keep_prob, dp_output_keep_prob = 1.0, 1.0
      last_input_keep_prob, last_output_keep_prob = 1.0, 1.0
      emb_keep_prob, recurrent_keep_prob = 1.0, 1.0
      input_weight_keep_prob, recurrent_weight_keep_prob = 1.0, 1.0


    self._output_layer = tf.layers.Dense(
      self._fc_dim, 
      kernel_regularizer=regularizer,
      kernel_initializer=initializer,
      use_bias=fc_use_bias,
      dtype=self._params['dtype']
    )

    if self._weight_tied:
      last_cell_params = copy.deepcopy(self.params['core_cell_params'])
      last_cell_params['num_units'] = self._emb_size
    else:
      last_cell_params = self.params['core_cell_params']
    
    last_output_dim = last_cell_params['num_units']

    if self._use_cell_state:
      last_output_dim = 2 * last_output_dim


    fake_input = tf.zeros(shape=(1, last_output_dim), 
                          dtype=self._params['dtype'])
    fake_output = self._output_layer.apply(fake_input)
    with tf.variable_scope("dense", reuse=True):
      dense_weights = tf.get_variable("kernel")
      dense_biases = tf.get_variable("bias")
    
    if self._weight_tied and self._lm_phase:
      enc_emb_w = tf.transpose(dense_weights)
    else:
      enc_emb_w = tf.get_variable(
        name="EncoderEmbeddingMatrix",
        shape=[self._vocab_size, self._emb_size],
        dtype=self._params['dtype']
      )

    self._enc_emb_w = tf.nn.dropout(enc_emb_w, keep_prob=emb_keep_prob)

    if use_cudnn_rnn:
      if self._mode == 'train' or self._mode == 'eval':
        all_cudnn_classes = [
            i[1]
            for i in inspect.getmembers(tf.contrib.cudnn_rnn, inspect.isclass)
        ]

        if not cudnn_rnn_type in all_cudnn_classes:
          raise TypeError("rnn_type must be a Cudnn RNN class")

        rnn_block = cudnn_rnn_type(
            num_layers=self.params['encoder_layers'],
            num_units=self._emb_size, 
            dtype=self._params['dtype'],
            name="cudnn_rnn"
        )
      else:
        # Transferring weights from model trained with CudnnLSTM/CudnnGRU
        # to CudnnCompatibleLSTMCell/CudnnCompatibleGRUCell for inference
        if 'CudnnLSTM' in str(cudnn_rnn_type):
          cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=self._emb_size)
        elif 'CudnnGRU' in str(cudnn_rnn_type):
          cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units=self._emb_size)

        fwd_cells = [cell() for _ in range(self.params['encoder_layers'])]
        self._encoder_cell_fw = tf.nn.rnn_cell.MultiRNNCell(fwd_cells)
    else:
      fwd_cells = [
        single_cell(cell_class=self.params['core_cell'],
                    cell_params=self.params['core_cell_params'],
                    dp_input_keep_prob=dp_input_keep_prob,
                    dp_output_keep_prob=dp_output_keep_prob,
                    recurrent_keep_prob=recurrent_keep_prob,
                    input_weight_keep_prob=input_weight_keep_prob,
                    recurrent_weight_keep_prob=recurrent_weight_keep_prob,
                    weight_variational=self.params['weight_variational'],
                    dropout_seed=self.params['dropout_seed'],
                    residual_connections=self.params['encoder_use_skip_connections'],
                    awd_initializer=self.params['awd_initializer'],
                    dtype=self._params['dtype']
                    ) for _ in range(self.params['encoder_layers'] - 1)]

      fwd_cells.append(
        single_cell(cell_class=self.params['core_cell'],
                    cell_params=last_cell_params,
                    dp_input_keep_prob=last_input_keep_prob,
                    dp_output_keep_prob=last_output_keep_prob,
                    recurrent_keep_prob=recurrent_keep_prob,
                    input_weight_keep_prob=input_weight_keep_prob,
                    recurrent_weight_keep_prob=recurrent_weight_keep_prob,
                    weight_variational=self.params['weight_variational'],
                    dropout_seed=self.params['dropout_seed'],
                    residual_connections=self.params['encoder_use_skip_connections'],
                    awd_initializer=self.params['awd_initializer'],
                    dtype=self._params['dtype']
                    )
        )

      self._encoder_cell_fw = tf.contrib.rnn.MultiRNNCell(fwd_cells)

    time_major = self.params.get("time_major", False)
    use_swap_memory = self.params.get("use_swap_memory", False)

    source_sequence = input_dict['source_tensors'][0]
    source_length = input_dict['source_tensors'][1]

    # Inference for language modeling requires a different graph
    if (not self._lm_phase) or self._mode == 'train' or self._mode == 'eval':
      embedded_inputs = tf.cast(tf.nn.embedding_lookup(
        self.enc_emb_w,
        source_sequence,
      ), self.params['dtype'])

      if use_cudnn_rnn:
        # The CudnnLSTM will return encoder_state as a tuple of hidden 
        # and cell values that. The hidden and cell tensors are stored for
        # each LSTM Layer.

        # reshape to [B, T, C] --> [T, B, C]
        if time_major == False:
          embedded_inputs = tf.transpose(embedded_inputs, [1, 0, 2])

        rnn_block.build(embedded_inputs.get_shape())
        encoder_outputs, encoder_state = rnn_block(embedded_inputs)
        encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])
      else:
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
          cell=self._encoder_cell_fw,
          inputs=embedded_inputs,
          sequence_length=source_length,
          time_major=time_major,
          swap_memory=use_swap_memory,
          dtype=self._params['dtype'],
          scope='decoder',
        )

      if not self._lm_phase:
        # CudnnLSTM stores cell and hidden state differently
        if use_cudnn_rnn:
          if self._use_cell_state:
            encoder_outputs = tf.concat([encoder_state[0][-1], encoder_state[1][-1]], axis=1)
          else:
            encoder_outputs = encoder_state[0][-1]
        else:
          if self._use_cell_state:
            encoder_outputs = tf.concat([encoder_state[-1].h, encoder_state[-1].c], axis=1)
          else:
            encoder_outputs = encoder_state[-1].h

      if self._mode == 'train' and self._num_sampled < self._fc_dim: # sampled softmax
        output_dict = {'weights': enc_emb_w,
                    'bias': dense_biases,
                    'inputs': encoder_outputs,
                    'logits': encoder_outputs,
                    'outputs': [encoder_outputs],
                    'num_sampled': self._num_sampled}
      else: # full softmax
        logits = self._output_layer.apply(encoder_outputs)
        output_dict = {'logits': logits, 'outputs': [logits]}
    else: # infer in LM phase
      # This portion of graph is required to restore weights from CudnnLSTM to 
      # CudnnCompatibleLSTMCell/CudnnCompatibleGRUCell
      if use_cudnn_rnn:
        embedded_inputs = tf.cast(tf.nn.embedding_lookup(
          self.enc_emb_w,
          source_sequence,
        ), self.params['dtype'])

        # Scope must remain unset to restore weights
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell=self._encoder_cell_fw,
            inputs=embedded_inputs,
            sequence_length=source_length,
            time_major=time_major,
            swap_memory=use_swap_memory,
            dtype=self._params['dtype']
        )

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
          batch_size=self._batch_size, dtype=self._params['dtype'],
        ),
        output_layer=self._output_layer,
      )
      maximum_iterations = tf.constant(self._num_tokens_gen)

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

    return output_dict

  @property
  def vocab_size(self):
    return self._vocab_size

  @property
  def emb_size(self):
    return self._emb_size

  @property
  def enc_emb_w(self):
    return self._enc_emb_w