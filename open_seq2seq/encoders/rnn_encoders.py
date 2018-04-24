# Copyright (c) 2018 NVIDIA Corporation
"""
RNN-based encoders
"""
from __future__ import absolute_import, division, print_function
import copy
import tensorflow as tf

from open_seq2seq.parts.utils import create_rnn_cell
from .encoder import Encoder


class UnidirectionalRNNEncoderWithEmbedding(Encoder):
  """
  Uni-directional RNN decoder with embeddings.
  Can support various RNN cell types.
  """
  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
      'src_vocab_size': int,
      'src_emb_size': int,
      'encoder_cell_units': int,
      'encoder_cell_type': ['lstm', 'gru', 'glstm', 'slstm'],
      'encoder_layers': int,
      'encoder_use_skip_connections': bool,
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
               name="unidir_rnn_encoder_with_emb", mode='train'):
    """
    Initializes uni-directional encoder with embeddings
    :param params: dictionary with encoder parameters
    Must define:
      * src_vocab_size - data vocabulary size
      * src_emb_size - size of embedding to use
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
    """
    super(UnidirectionalRNNEncoderWithEmbedding, self).__init__(
      params,
      model,
      name=name,
      mode=mode,
    )

    self._src_vocab_size = self.params['src_vocab_size']
    self._src_emb_size = self.params['src_emb_size']

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
    # TODO: make a separate level of config for cell_params?
    cell_params = copy.deepcopy(self.params)
    cell_params["num_units"] = self.params['encoder_cell_units']

    self._enc_emb_w = tf.get_variable(
      name="EncoderEmbeddingMatrix",
      shape=[self._src_vocab_size, self._src_emb_size],
      dtype=tf.float32
    )

    if self._mode == "train":
      dp_input_keep_prob = self.params['encoder_dp_input_keep_prob']
      dp_output_keep_prob = self.params['encoder_dp_output_keep_prob']
    else:
      dp_input_keep_prob = 1.0
      dp_output_keep_prob = 1.0

    self._encoder_cell_fw = create_rnn_cell(
      cell_type=self.params['encoder_cell_type'],
      cell_params=cell_params,
      num_layers=self.params['encoder_layers'],
      dp_input_keep_prob=dp_input_keep_prob,
      dp_output_keep_prob=dp_output_keep_prob,
      residual_connections=self.params['encoder_use_skip_connections'],
    )

    time_major = self.params.get("time_major", False)
    use_swap_memory = self.params.get("use_swap_memory", False)

    embedded_inputs = tf.cast(tf.nn.embedding_lookup(
      self.enc_emb_w,
      input_dict['src_sequence'],
    ), self.params['dtype'])

    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
      cell=self._encoder_cell_fw,
      inputs=embedded_inputs,
      sequence_length=input_dict['src_length'],
      time_major=time_major,
      swap_memory=use_swap_memory,
      dtype=embedded_inputs.dtype,
    )
    return {'outputs': encoder_outputs,
            'state': encoder_state,
            'src_lengths': input_dict['src_length'],
            'encoder_input': input_dict['src_sequence']}

  @property
  def src_vocab_size(self):
    return self._src_vocab_size

  @property
  def src_emb_size(self):
    return self._src_emb_size

  @property
  def enc_emb_w(self):
    return self._enc_emb_w


class BidirectionalRNNEncoderWithEmbedding(Encoder):
  """
  Bi-directional RNN-based encoder with embeddings.
  Can support various RNN cell types.
  """
  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
      'src_vocab_size': int,
      'src_emb_size': int,
      'encoder_cell_units': int,
      'encoder_cell_type': ['lstm', 'gru', 'glstm', 'slstm'],
      'encoder_layers': int,
      'encoder_use_skip_connections': bool,
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
               name="bidir_rnn_encoder_with_emb", mode='train'):
    """
    Initializes bi-directional encoder with embeddings
    :param params: dictionary with encoder parameters
    Must define:
      * src_vocab_size - data vocabulary size
      * src_emb_size - size of embedding to use
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
    super(BidirectionalRNNEncoderWithEmbedding, self).__init__(
      params, model, name=name, mode=mode,
    )

    self._src_vocab_size = self.params['src_vocab_size']
    self._src_emb_size = self.params['src_emb_size']

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

    self._enc_emb_w = tf.get_variable(
      name="EncoderEmbeddingMatrix",
      shape=[self._src_vocab_size, self._src_emb_size],
      dtype=tf.float32
    )

    cell_params = copy.deepcopy(self.params)
    cell_params["num_units"] = self.params['encoder_cell_units']

    if self._mode == "train":
      dp_input_keep_prob = self.params['encoder_dp_input_keep_prob']
      dp_output_keep_prob = self.params['encoder_dp_output_keep_prob']
    else:
      dp_input_keep_prob = 1.0
      dp_output_keep_prob = 1.0

    with tf.variable_scope("FW"):
      self._encoder_cell_fw = create_rnn_cell(
        cell_type=self.params['encoder_cell_type'],
        cell_params=cell_params,
        num_layers=self.params['encoder_layers'],
        dp_input_keep_prob=dp_input_keep_prob,
        dp_output_keep_prob=dp_output_keep_prob,
        residual_connections=self.params['encoder_use_skip_connections']
      )

    with tf.variable_scope("BW"):
      self._encoder_cell_bw = create_rnn_cell(
        cell_type=self.params['encoder_cell_type'],
        cell_params=cell_params,
        num_layers=self.params['encoder_layers'],
        dp_input_keep_prob=dp_input_keep_prob,
        dp_output_keep_prob=dp_output_keep_prob,
        residual_connections=self.params['encoder_use_skip_connections']
      )

    embedded_inputs = tf.cast(tf.nn.embedding_lookup(
      self.enc_emb_w,
      input_dict['src_sequence'],
    ), self.params['dtype'])

    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=self._encoder_cell_fw,
      cell_bw=self._encoder_cell_bw,
      inputs=embedded_inputs,
      sequence_length=input_dict['src_length'],
      time_major=time_major,
      swap_memory=use_swap_memory,
      dtype=embedded_inputs.dtype,
    )
    encoder_outputs = tf.concat(encoder_output, 2)
    return {'outputs': encoder_outputs,
            'state': encoder_state,
            'src_lengths': input_dict['src_length'],
            'encoder_input': input_dict['src_sequence']}

  @property
  def src_vocab_size(self):
    return self._src_vocab_size

  @property
  def src_emb_size(self):
    return self._src_emb_size

  @property
  def enc_emb_w(self):
    return self._enc_emb_w


class GNMTLikeEncoderWithEmbedding(Encoder):
  """
  Encoder similar to the one used in
  GNMT model: https://arxiv.org/abs/1609.08144.
  Must have at least 2 layers
  """
  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
      'src_vocab_size': int,
      'src_emb_size': int,
      'encoder_cell_units': int,
      'encoder_cell_type': ['lstm', 'gru', 'glstm', 'slstm'],
      'encoder_layers': int,
      'encoder_use_skip_connections': bool,
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
               name="gnmt_encoder_with_emb", mode='train'):
    """
    Encodes data into representation
    :param params: a Python dictionary.
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
    super(GNMTLikeEncoderWithEmbedding, self).__init__(
      params, model, name=name, mode=mode,
    )

    self._src_vocab_size = self.params['src_vocab_size']
    self._src_emb_size = self.params['src_emb_size']

  def _encode(self, input_dict):

    self._enc_emb_w = tf.get_variable(
      name="EncoderEmbeddingMatrix",
      shape=[self._src_vocab_size, self._src_emb_size],
      dtype=tf.float32
    )

    if self.params['encoder_layers'] < 2:
      raise ValueError("GNMT encoder must have at least 2 layers")

    cell_params = copy.deepcopy(self.params)
    cell_params["num_units"] = self.params['encoder_cell_units']

    with tf.variable_scope("Level1FW"):
      self._encoder_l1_cell_fw = create_rnn_cell(
        cell_type=self.params['encoder_cell_type'],
        cell_params=cell_params,
        num_layers=1,
        dp_input_keep_prob=1.0,
        dp_output_keep_prob=1.0,
        residual_connections=False,
      )
    with tf.variable_scope("Level1BW"):
      self._encoder_l1_cell_bw = create_rnn_cell(
        cell_type=self.params['encoder_cell_type'],
        cell_params=cell_params,
        num_layers=1,
        dp_input_keep_prob=1.0,
        dp_output_keep_prob=1.0,
        residual_connections=False,
      )

    if self._mode == "train":
      dp_input_keep_prob = self.params['encoder_dp_input_keep_prob']
      dp_output_keep_prob = self.params['encoder_dp_output_keep_prob']
    else:
      dp_input_keep_prob = 1.0
      dp_output_keep_prob = 1.0

    with tf.variable_scope("UniDirLevel"):
      self._encoder_cells = create_rnn_cell(
        cell_type=self.params['encoder_cell_type'],
        cell_params=cell_params,
        num_layers=self.params['encoder_layers'] - 1,
        dp_input_keep_prob=dp_input_keep_prob,
        dp_output_keep_prob=dp_output_keep_prob,
        residual_connections=False,
        wrap_to_multi_rnn=False,
      )
      # add residual connections starting from the third layer
      for idx, cell in enumerate(self._encoder_cells):
        if idx > 0:
          self._encoder_cells[idx] = tf.contrib.rnn.ResidualWrapper(cell)

    time_major = self.params.get("time_major", False)
    use_swap_memory = self.params.get("use_swap_memory", False)
    embedded_inputs = tf.cast(tf.nn.embedding_lookup(
      self.enc_emb_w,
      input_dict['src_sequence'],
    ), self.params['dtype'])

    # first bi-directional layer
    _encoder_output, _ = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=self._encoder_l1_cell_fw,
      cell_bw=self._encoder_l1_cell_bw,
      inputs=embedded_inputs,
      sequence_length=input_dict['src_length'],
      swap_memory=use_swap_memory,
      time_major=time_major,
      dtype=embedded_inputs.dtype,
    )
    encoder_l1_outputs = tf.concat(_encoder_output, 2)

    # stack of unidirectional layers
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
      cell=tf.contrib.rnn.MultiRNNCell(self._encoder_cells),
      inputs=encoder_l1_outputs,
      sequence_length=input_dict['src_length'],
      swap_memory=use_swap_memory,
      time_major = time_major,
      dtype=encoder_l1_outputs.dtype,
    )

    return {'outputs': encoder_outputs,
            'state': encoder_state,
            'src_lengths': input_dict['src_length'],
            'encoder_input': input_dict['src_sequence']}

  @property
  def src_vocab_size(self):
    return self._src_vocab_size

  @property
  def src_emb_size(self):
    return self._src_emb_size

  @property
  def enc_emb_w(self):
    return self._enc_emb_w
