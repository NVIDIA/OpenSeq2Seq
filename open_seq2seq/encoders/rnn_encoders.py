# Copyright (c) 2018 NVIDIA Corporation
"""
RNN-based encoders
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

from open_seq2seq.parts.rnns.utils import single_cell
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
        'core_cell': None,
        'core_cell_params': dict,
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
    """Initializes uni-directional encoder with embeddings.

    Args:
       params (dict): dictionary with encoder parameters
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

    self._enc_emb_w = None
    self._encoder_cell_fw = None

  def _encode(self, input_dict):
    """Encodes data into representation.

    Args:
      input_dict: a Python dictionary.
        Must define:
          * src_inputs - a Tensor of shape [batch_size, time] or
                         [time, batch_size]
                         (depending on time_major param)
          * src_lengths - a Tensor of shape [batch_size]

    Returns:
       a Python dictionary with:
      * encoder_outputs - a Tensor of shape
                          [batch_size, time, representation_dim]
      or [time, batch_size, representation_dim]
      * encoder_state - a Tensor of shape [batch_size, dim]
      * src_lengths - (copy ref from input) a Tensor of shape [batch_size]
    """
    # TODO: make a separate level of config for cell_params?
    source_sequence = input_dict['source_tensors'][0]
    source_length = input_dict['source_tensors'][1]

    self._enc_emb_w = tf.get_variable(
        name="EncoderEmbeddingMatrix",
        shape=[self._src_vocab_size, self._src_emb_size],
        dtype=tf.float32,
    )

    if self._mode == "train":
      dp_input_keep_prob = self.params['encoder_dp_input_keep_prob']
      dp_output_keep_prob = self.params['encoder_dp_output_keep_prob']
    else:
      dp_input_keep_prob = 1.0
      dp_output_keep_prob = 1.0

    fwd_cells = [
        single_cell(
            cell_class=self.params['core_cell'],
            cell_params=self.params.get('core_cell_params', {}),
            dp_input_keep_prob=dp_input_keep_prob,
            dp_output_keep_prob=dp_output_keep_prob,
            residual_connections=self.params['encoder_use_skip_connections']
        ) for _ in range(self.params['encoder_layers'])
    ]
    # pylint: disable=no-member
    self._encoder_cell_fw = tf.contrib.rnn.MultiRNNCell(fwd_cells)

    time_major = self.params.get("time_major", False)
    use_swap_memory = self.params.get("use_swap_memory", False)

    embedded_inputs = tf.cast(
        tf.nn.embedding_lookup(
            self.enc_emb_w,
            source_sequence,
        ),
        self.params['dtype'],
    )

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
        'encoder_layers': int,
        'encoder_use_skip_connections': bool,
        'core_cell': None,
        'core_cell_params': dict,
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
    """Initializes bi-directional encoder with embeddings.

    Args:
      params (dict): dictionary with encoder parameters
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

    Returns:
      encoder_params
    """
    super(BidirectionalRNNEncoderWithEmbedding, self).__init__(
        params, model, name=name, mode=mode,
    )

    self._src_vocab_size = self.params['src_vocab_size']
    self._src_emb_size = self.params['src_emb_size']

    self._enc_emb_w = None
    self._encoder_cell_fw = None
    self._encoder_cell_bw = None

  def _encode(self, input_dict):
    """Encodes data into representation.
    Args:
      input_dict: a Python dictionary.
        Must define:
          *src_inputs - a Tensor of shape [batch_size, time] or
                        [time, batch_size]
                        (depending on time_major param)
          * src_lengths - a Tensor of shape [batch_size]

    Returns:
      a Python dictionary with:
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
        shape=[self._src_vocab_size, self._src_emb_size],
        dtype=tf.float32
    )

    if self._mode == "train":
      dp_input_keep_prob = self.params['encoder_dp_input_keep_prob']
      dp_output_keep_prob = self.params['encoder_dp_output_keep_prob']
    else:
      dp_input_keep_prob = 1.0
      dp_output_keep_prob = 1.0

    fwd_cells = [
        single_cell(
            cell_class=self.params['core_cell'],
            cell_params=self.params.get('core_cell_params', {}),
            dp_input_keep_prob=dp_input_keep_prob,
            dp_output_keep_prob=dp_output_keep_prob,
            residual_connections=self.params['encoder_use_skip_connections'],
        ) for _ in range(self.params['encoder_layers'])
    ]
    bwd_cells = [
        single_cell(
            cell_class=self.params['core_cell'],
            cell_params=self.params.get('core_cell_params', {}),
            dp_input_keep_prob=dp_input_keep_prob,
            dp_output_keep_prob=dp_output_keep_prob,
            residual_connections=self.params['encoder_use_skip_connections'],
        ) for _ in range(self.params['encoder_layers'])
    ]

    with tf.variable_scope("FW"):
      # pylint: disable=no-member
      self._encoder_cell_fw = tf.contrib.rnn.MultiRNNCell(fwd_cells)

    with tf.variable_scope("BW"):
      # pylint: disable=no-member
      self._encoder_cell_bw = tf.contrib.rnn.MultiRNNCell(bwd_cells)

    embedded_inputs = tf.cast(
        tf.nn.embedding_lookup(
            self.enc_emb_w,
            source_sequence,
        ),
        self.params['dtype']
    )

    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=self._encoder_cell_fw,
        cell_bw=self._encoder_cell_bw,
        inputs=embedded_inputs,
        sequence_length=source_length,
        time_major=time_major,
        swap_memory=use_swap_memory,
        dtype=embedded_inputs.dtype,
    )
    encoder_outputs = tf.concat(encoder_output, 2)
    return {'outputs': encoder_outputs,
            'state': encoder_state,
            'src_lengths': source_length,
            'encoder_input': source_sequence}

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
        'core_cell': None,
        'core_cell_params': dict,
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
    """Encodes data into representation.

    Args:
      params (dict): a Python dictionary.
        Must define:
          * src_inputs - a Tensor of shape [batch_size, time] or
                         [time, batch_size]
                         (depending on time_major param)
          * src_lengths - a Tensor of shape [batch_size]

    Returns:
      a Python dictionary with:
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

    self._encoder_l1_cell_fw = None
    self._encoder_l1_cell_bw = None
    self._encoder_cells = None
    self._enc_emb_w = None

  def _encode(self, input_dict):
    source_sequence = input_dict['source_tensors'][0]
    source_length = input_dict['source_tensors'][1]
    self._enc_emb_w = tf.get_variable(
        name="EncoderEmbeddingMatrix",
        shape=[self._src_vocab_size, self._src_emb_size],
        dtype=tf.float32,
    )

    if self.params['encoder_layers'] < 2:
      raise ValueError("GNMT encoder must have at least 2 layers")

    with tf.variable_scope("Level1FW"):
      self._encoder_l1_cell_fw = single_cell(
          cell_class=self.params['core_cell'],
          cell_params=self.params.get('core_cell_params', {}),
          dp_input_keep_prob=1.0,
          dp_output_keep_prob=1.0,
          residual_connections=False,
      )

    with tf.variable_scope("Level1BW"):
      self._encoder_l1_cell_bw = single_cell(
          cell_class=self.params['core_cell'],
          cell_params=self.params.get('core_cell_params', {}),
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
      self._encoder_cells = [
          single_cell(
              cell_class=self.params['core_cell'],
              cell_params=self.params.get('core_cell_params', {}),
              dp_input_keep_prob=dp_input_keep_prob,
              dp_output_keep_prob=dp_output_keep_prob,
              residual_connections=False,
          ) for _ in range(self.params['encoder_layers'] - 1)
      ]

      # add residual connections starting from the third layer
      for idx, cell in enumerate(self._encoder_cells):
        if idx > 0:
          # pylint: disable=no-member
          self._encoder_cells[idx] = tf.contrib.rnn.ResidualWrapper(cell)

    time_major = self.params.get("time_major", False)
    use_swap_memory = self.params.get("use_swap_memory", False)
    embedded_inputs = tf.cast(
        tf.nn.embedding_lookup(
            self.enc_emb_w,
            source_sequence,
        ),
        self.params['dtype'],
    )

    # first bi-directional layer
    _encoder_output, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=self._encoder_l1_cell_fw,
        cell_bw=self._encoder_l1_cell_bw,
        inputs=embedded_inputs,
        sequence_length=source_length,
        swap_memory=use_swap_memory,
        time_major=time_major,
        dtype=embedded_inputs.dtype,
    )
    encoder_l1_outputs = tf.concat(_encoder_output, 2)

    # stack of unidirectional layers
    # pylint: disable=no-member
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        cell=tf.contrib.rnn.MultiRNNCell(self._encoder_cells),
        inputs=encoder_l1_outputs,
        sequence_length=source_length,
        swap_memory=use_swap_memory,
        time_major=time_major,
        dtype=encoder_l1_outputs.dtype,
    )

    return {'outputs': encoder_outputs,
            'state': encoder_state,
            'src_lengths': source_length,
            'encoder_input': source_sequence}

  @property
  def src_vocab_size(self):
    return self._src_vocab_size

  @property
  def src_emb_size(self):
    return self._src_emb_size

  @property
  def enc_emb_w(self):
    return self._enc_emb_w


class GNMTLikeEncoderWithEmbedding_cuDNN(Encoder):
  """
    Encoder similar to the one used in
    GNMT model: https://arxiv.org/abs/1609.08144.
    Must have at least 2 layers. Uses cuDNN RNN blocks for efficiency
    """

  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
        'src_vocab_size': int,
        'src_emb_size': int,
        'encoder_cell_units': int,
        'encoder_cell_type': ['lstm', 'gru'],
        'encoder_layers': int,
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
        'encoder_dp_output_keep_prob': float,
    })

  def __init__(self, params, model,
               name="gnmt_encoder_with_emb_cudnn", mode='train'):
    """Encodes data into representation

    Args:
      params (dict): a Python dictionary.
        Must define:
          * src_inputs - a Tensor of shape [batch_size, time] or
                         [time, batch_size]
                         (depending on time_major param)
          * src_lengths - a Tensor of shape [batch_size]

    Returns:
      a Python dictionary with:
      * encoder_outputs - a Tensor of shape
                          [batch_size, time, representation_dim]
      or [time, batch_size, representation_dim]
      * encoder_state - a Tensor of shape [batch_size, dim]
      * src_lengths - (copy ref from input) a Tensor of shape [batch_size]
    """
    super(GNMTLikeEncoderWithEmbedding_cuDNN, self).__init__(
        params, model, name=name, mode=mode,
    )

    self._src_vocab_size = self.params['src_vocab_size']
    self._src_emb_size = self.params['src_emb_size']

    self._enc_emb_w = None

  def _encode(self, input_dict):
    source_sequence = input_dict['source_tensors'][0]
    source_length = input_dict['source_tensors'][1]
    self._enc_emb_w = tf.get_variable(
        name="EncoderEmbeddingMatrix",
        shape=[self._src_vocab_size, self._src_emb_size],
        dtype=tf.float32
    )

    if self.params['encoder_layers'] < 2:
      raise ValueError("GNMT encoder must have at least 2 layers")

    if self._mode == "train":
      dp_output_keep_prob = self.params['encoder_dp_output_keep_prob']
    else:
      dp_output_keep_prob = 1.0

    # source_sequence is of [batch, time] shape
    embedded_inputs = tf.cast(
        tf.nn.embedding_lookup(
            self.enc_emb_w,
            tf.transpose(source_sequence), # cudnn wants [time, batch, ...]
        ),
        self.params['dtype'],
    )

    with tf.variable_scope("Bi_Directional_Layer"):
      direction = cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION
      if self.params['encoder_cell_type'] == "gru":
        # pylint: disable=no-member
        bidirectional_block = tf.contrib.cudnn_rnn.CudnnGRU(
            num_layers=1,
            num_units=self.params['encoder_cell_units'],
            direction=direction,
            dropout=0.0,
            dtype=self.params['dtype'],
            name="cudnn_gru_bidi",
        )
      elif self.params['encoder_cell_type'] == "lstm":
        # pylint: disable=no-member
        bidirectional_block = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=1,
            num_units=self.params['encoder_cell_units'],
            direction=direction,
            dropout=0.0,
            dtype=self.params['dtype'],
            name="cudnn_lstm_bidi",
        )
      else:
        raise ValueError(
            "{} is not a valid rnn_type for cudnn_rnn layers".format(
                self.params['encoder_cell_units']
            )
        )
      bidi_output, bidi_state = bidirectional_block(embedded_inputs)

    with tf.variable_scope("Uni_Directional_Layer"):
      direction = cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION
      layer_input = bidi_output
      for ind in range(self.params['encoder_layers'] - 1):
        with tf.variable_scope("uni_layer_{}".format(ind)):
          if self.params['encoder_cell_type'] == "gru":
            # pylint: disable=no-member
            unidirectional_block = tf.contrib.cudnn_rnn.CudnnGRU(
                num_layers=1,
                num_units=self.params['encoder_cell_units'],
                direction=direction,
                dropout=1.0 - dp_output_keep_prob,
                dtype=self.params['dtype'],
                name="cudnn_gru_uni_{}".format(ind),
            )
          elif self.params['encoder_cell_type'] == "lstm":
            # pylint: disable=no-member
            unidirectional_block = tf.contrib.cudnn_rnn.CudnnLSTM(
                num_layers=1,
                num_units=self.params['encoder_cell_units'],
                direction=direction,
                dropout=1.0 - dp_output_keep_prob,
                dtype=self.params['dtype'],
                name="cudnn_lstm_uni_{}".format(ind),
            )
          layer_output, encoder_state = unidirectional_block(layer_input)
          if ind > 0:  # add residual connection
            layer_output = layer_input + layer_output
          layer_input = layer_output

    return {'outputs': tf.transpose(layer_input, perm=[1, 0, 2]),
            'state': None,
            'src_lengths': source_length,
            'encoder_input': source_sequence}

  @property
  def src_vocab_size(self):
    return self._src_vocab_size

  @property
  def src_emb_size(self):
    return self._src_emb_size

  @property
  def enc_emb_w(self):
    return self._enc_emb_w
