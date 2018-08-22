# Copyright (c) 2018 NVIDIA Corporation
"""
RNN-based decoders.
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import copy

import tensorflow as tf

from open_seq2seq.parts.rnns.attention_wrapper import BahdanauAttention, \
                                                      LuongAttention, \
                                                      AttentionWrapper
from open_seq2seq.parts.rnns.gnmt import GNMTAttentionMultiCell, \
                                         gnmt_residual_fn
from open_seq2seq.parts.rnns.rnn_beam_search_decoder import BeamSearchDecoder
from open_seq2seq.parts.rnns.utils import single_cell
from .decoder import Decoder


class RNNDecoderWithAttention(Decoder):
  """Typical RNN decoder with attention mechanism.
  """
  @staticmethod
  def get_required_params():
    return dict(Decoder.get_required_params(), **{
        'GO_SYMBOL': int,  # symbol id
        'END_SYMBOL': int,  # symbol id
        'tgt_vocab_size': int,
        'tgt_emb_size': int,
        'attention_layer_size': int,
        'attention_type': ['bahdanau', 'luong', 'gnmt', 'gnmt_v2'],
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
        'weight_tied': bool,
    })

  def __init__(self, params, model,
               name='rnn_decoder_with_attention', mode='train'):
    """Initializes RNN decoder with embedding.

    See parent class for arguments description.

    Config parameters:

    * **batch_size** (int) --- batch size.
    * **GO_SYMBOL** (int) --- GO symbol id, must be the same as used in
      data layer.
    * **END_SYMBOL** (int) --- END symbol id, must be the same as used in
      data layer.
    * **tgt_emb_size** (int) --- embedding size to use.
    * **core_cell_params** (dict) - parameters for RNN class
    * **core_cell** (string) - RNN class.
    * **decoder_dp_input_keep_prob** (float) - dropout input keep probability.
    * **decoder_dp_output_keep_prob** (float) - dropout output keep probability.
    * **decoder_use_skip_connections** (bool) - use residual connections or not.
    * **attention_type** (string) - bahdanau, luong, gnmt or gnmt_v2.
    * **bahdanau_normalize** (bool, optional) - whether to use normalization in
      bahdanau attention.
    * **luong_scale** (bool, optional) - whether to use scale in luong attention
    * ... add any cell-specific parameters here as well.
    """
    super(RNNDecoderWithAttention, self).__init__(params, model, name, mode)
    self._batch_size = self.params['batch_size']
    self.GO_SYMBOL = self.params['GO_SYMBOL']
    self.END_SYMBOL = self.params['END_SYMBOL']
    self._tgt_vocab_size = self.params['tgt_vocab_size']
    self._tgt_emb_size = self.params['tgt_emb_size']
    self._weight_tied = self.params.get('weight_tied', False)

  def _build_attention(self,
                       encoder_outputs,
                       encoder_sequence_length):
    """Builds Attention part of the graph.
    Currently supports "bahdanau" and "luong".
    """
    with tf.variable_scope("AttentionMechanism"):
      attention_depth = self.params['attention_layer_size']
      if self.params['attention_type'] == 'bahdanau':
        if 'bahdanau_normalize' in self.params:
          bah_normalize = self.params['bahdanau_normalize']
        else:
          bah_normalize = False
        attention_mechanism = BahdanauAttention(
            num_units=attention_depth,
            memory=encoder_outputs,
            normalize=bah_normalize,
            memory_sequence_length=encoder_sequence_length,
            probability_fn=tf.nn.softmax,
            dtype=tf.get_variable_scope().dtype
        )
      elif self.params['attention_type'] == 'luong':
        if 'luong_scale' in self.params:
          luong_scale = self.params['luong_scale']
        else:
          luong_scale = False
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
        cells[idx] = tf.contrib.rnn.ResidualWrapper(  # pylint: disable=no-member
            cell,
            residual_fn=gnmt_residual_fn,
        )
    return cells

  def _decode(self, input_dict):
    """Decodes representation into data.

    Args:
      input_dict (dict): Python dictionary with inputs to decoder.


    Config parameters:

    * **src_inputs** --- Decoder input Tensor of shape [batch_size, time, dim]
      or [time, batch_size, dim]
    * **src_lengths** --- Decoder input lengths Tensor of shape [batch_size]
    * **tgt_inputs** --- Only during training. labels Tensor of the
      shape [batch_size, time] or [time, batch_size].
    * **tgt_lengths** --- Only during training. labels lengths
      Tensor of the shape [batch_size].

    Returns:
      dict: Python dictionary with:
      * final_outputs - tensor of shape [batch_size, time, dim]
                        or [time, batch_size, dim]
      * final_state - tensor with decoder final state
      * final_sequence_lengths - tensor of shape [batch_size, time]
                                 or [time, batch_size]
    """
    encoder_outputs = input_dict['encoder_output']['outputs']
    enc_src_lengths = input_dict['encoder_output']['src_lengths']
    tgt_inputs = input_dict['target_tensors'][0] if 'target_tensors' in \
                                                    input_dict else None
    tgt_lengths = input_dict['target_tensors'][1] if 'target_tensors' in \
                                                     input_dict else None

    self._output_projection_layer = tf.layers.Dense(
        self._tgt_vocab_size, use_bias=False,
    )

    if not self._weight_tied:
      self._dec_emb_w = tf.get_variable(
          name='DecoderEmbeddingMatrix',
          shape=[self._tgt_vocab_size, self._tgt_emb_size],
          dtype=tf.float32
      )
    else:
      fake_input = tf.zeros(shape=(1, self._tgt_emb_size))
      fake_output = self._output_projection_layer.apply(fake_input)
      with tf.variable_scope("dense", reuse=True):
        dense_weights = tf.get_variable("kernel")
        self._dec_emb_w = tf.transpose(dense_weights)

    if self._mode == "train":
      dp_input_keep_prob = self.params['decoder_dp_input_keep_prob']
      dp_output_keep_prob = self.params['decoder_dp_output_keep_prob']
    else:
      dp_input_keep_prob = 1.0
      dp_output_keep_prob = 1.0

    residual_connections = self.params['decoder_use_skip_connections']

    # list of cells
    cell_params = self.params.get('core_cell_params', {})
    self._decoder_cells = [
        single_cell(
            cell_class=self.params['core_cell'],
            cell_params=self.params.get('core_cell_params', {}),
            dp_input_keep_prob=dp_input_keep_prob,
            dp_output_keep_prob=dp_output_keep_prob,
            # residual connections are added a little differently for GNMT
            residual_connections=False if self.params['attention_type'].startswith('gnmt')
                                 else residual_connections,
        ) for _ in range(self.params['decoder_layers'] - 1)
    ]

    last_cell_params = copy.deepcopy(cell_params)
    if self._weight_tied:
      last_cell_params['num_units'] = self._tgt_emb_size

    last_cell = single_cell(
            cell_class=self.params['core_cell'],
            cell_params=last_cell_params,
            dp_input_keep_prob=dp_input_keep_prob,
            dp_output_keep_prob=dp_output_keep_prob,
            # residual connections are added a little differently for GNMT
            residual_connections=False if self.params['attention_type'].startswith('gnmt')
                                 else residual_connections,
        )
    self._decoder_cells.append(last_cell)



    attention_mechanism = self._build_attention(
        encoder_outputs,
        enc_src_lengths,
    )
    if self.params['attention_type'].startswith('gnmt'):
      attention_cell = self._decoder_cells.pop(0)
      attention_cell = AttentionWrapper(
          attention_cell,
          attention_mechanism=attention_mechanism,
          attention_layer_size=None,
          output_attention=False,
          name="gnmt_attention",
      )
      attentive_decoder_cell = GNMTAttentionMultiCell(
          attention_cell,
          self._add_residual_wrapper(self._decoder_cells) if residual_connections else self._decoder_cells,
          use_new_attention=(self.params['attention_type'] == 'gnmt_v2'),
      )
    else:
      attentive_decoder_cell = AttentionWrapper(
          # pylint: disable=no-member
          cell=tf.contrib.rnn.MultiRNNCell(self._decoder_cells),
          attention_mechanism=attention_mechanism,
      )
    if self._mode == "train":
      input_vectors = tf.cast(
        tf.nn.embedding_lookup(self._dec_emb_w, tgt_inputs),
        dtype=self.params['dtype'],
      )
      helper = tf.contrib.seq2seq.TrainingHelper(  # pylint: disable=no-member
          inputs=input_vectors,
          sequence_length=tgt_lengths,
      )
      decoder = tf.contrib.seq2seq.BasicDecoder(  # pylint: disable=no-member
          cell=attentive_decoder_cell,
          helper=helper,
          output_layer=self._output_projection_layer,
          initial_state=attentive_decoder_cell.zero_state(
              self._batch_size, dtype=encoder_outputs.dtype,
          ),
      )
    elif self._mode == "infer" or self._mode == "eval":
      embedding_fn = lambda ids: tf.cast(
          tf.nn.embedding_lookup(self._dec_emb_w, ids),
          dtype=self.params['dtype'],
      )
      # pylint: disable=no-member
      helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
          embedding=embedding_fn,
          start_tokens=tf.fill([self._batch_size], self.GO_SYMBOL),
          end_token=self.END_SYMBOL,
      )
      decoder = tf.contrib.seq2seq.BasicDecoder(  # pylint: disable=no-member
          cell=attentive_decoder_cell,
          helper=helper,
          initial_state=attentive_decoder_cell.zero_state(
              batch_size=self._batch_size, dtype=encoder_outputs.dtype,
          ),
          output_layer=self._output_projection_layer,
      )
    else:
      raise ValueError(
          "Unknown mode for decoder: {}".format(self._mode)
      )

    time_major = self.params.get("time_major", False)
    use_swap_memory = self.params.get("use_swap_memory", False)
    if self._mode == 'train':
      maximum_iterations = tf.reduce_max(tgt_lengths)
    else:
      maximum_iterations = tf.reduce_max(enc_src_lengths) * 2

    # pylint: disable=no-member
    final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        impute_finished=True,
        maximum_iterations=maximum_iterations,
        swap_memory=use_swap_memory,
        output_time_major=time_major,
    )

    return {'logits': final_outputs.rnn_output if not time_major else
            tf.transpose(final_outputs.rnn_output, perm=[1, 0, 2]),
            'outputs': [tf.argmax(final_outputs.rnn_output, axis=-1)],
            'final_state': final_state,
            'final_sequence_lengths': final_sequence_lengths}


class BeamSearchRNNDecoderWithAttention(RNNDecoderWithAttention):
  """
  Beam search version of RNN-based decoder with attention.
  Can be used only during Inference (mode=infer)
  """
  @staticmethod
  def get_optional_params():
    return dict(RNNDecoderWithAttention.get_optional_params(), **{
        'length_penalty': float,
        'beam_width': int,
    })

  def __init__(self, params, model,
               name="rnn_decoder_with_attention", mode='train'):
    """Initializes beam search decoder.

    Args:
      params(dict): dictionary with decoder parameters

    Config parameters:

    * **batch_size** --- batch size
    * **GO_SYMBOL** --- GO symbol id, must be the same as used in data layer
    * **END_SYMBOL** --- END symbol id, must be the same as used in data layer
    * **tgt_vocab_size** --- vocabulary size of target
    * **tgt_emb_size** --- embedding to use
    * **decoder_cell_units** --- number of units in RNN
    * **decoder_cell_type** --- RNN type: lstm, gru, glstm, etc.
    * **decoder_dp_input_keep_prob** ---
    * **decoder_dp_output_keep_prob** ---
    * **decoder_use_skip_connections** --- use residual connections or not
    * **attention_type** --- bahdanau, luong, gnmt, gnmt_v2
    * **bahdanau_normalize** --- (optional)
    * **luong_scale** --- (optional)
    * **mode** --- train or infer
    ... add any cell-specific parameters here as well
    """
    super(BeamSearchRNNDecoderWithAttention, self).__init__(
        params, model, name, mode,
    )
    if self._mode != 'infer':
      raise ValueError(
          'BeamSearch decoder only supports infer mode, but got {}'.format(
              self._mode,
          )
      )
    if "length_penalty" not in self.params:
      self._length_penalty_weight = 0.0
    else:
      self._length_penalty_weight = self.params["length_penalty"]
    # beam_width of 1 should be same as argmax decoder
    if "beam_width" not in self.params:
      self._beam_width = 1
    else:
      self._beam_width = self.params["beam_width"]


  def _decode(self, input_dict):
    """Decodes representation into data.

    Args:
      input_dict (dict): Python dictionary with inputs to decoder

    Must define:
      * src_inputs - decoder input Tensor of shape [batch_size, time, dim]
                     or [time, batch_size, dim]
      * src_lengths - decoder input lengths Tensor of shape [batch_size]
    Does not need tgt_inputs and tgt_lengths

    Returns:
      dict: a Python dictionary with:
      * final_outputs - tensor of shape [batch_size, time, dim] or
                        [time, batch_size, dim]
      * final_state - tensor with decoder final state
      * final_sequence_lengths - tensor of shape [batch_size, time] or
                                 [time, batch_size]
    """
    encoder_outputs = input_dict['encoder_output']['outputs']
    enc_src_lengths = input_dict['encoder_output']['src_lengths']

    

    self._output_projection_layer = tf.layers.Dense(
        self._tgt_vocab_size, use_bias=False,
    )

    if not self._weight_tied:
      self._dec_emb_w = tf.get_variable(
          name='DecoderEmbeddingMatrix',
          shape=[self._tgt_vocab_size, self._tgt_emb_size],
          dtype=tf.float32
      )
    else:
      fake_input = tf.zeros(shape=(1, self._tgt_emb_size))
      fake_output = self._output_projection_layer.apply(fake_input)
      with tf.variable_scope("dense", reuse=True):
        dense_weights = tf.get_variable("kernel")
        self._dec_emb_w = tf.transpose(dense_weights)



    if self._mode == "train":
      dp_input_keep_prob = self.params['decoder_dp_input_keep_prob']
      dp_output_keep_prob = self.params['decoder_dp_output_keep_prob']
    else:
      dp_input_keep_prob = 1.0
      dp_output_keep_prob = 1.0

    residual_connections = self.params['decoder_use_skip_connections']
    # list of cells
    cell_params = self.params.get('core_cell_params', {})
    

    self._decoder_cells = [
        single_cell(
            cell_class=self.params['core_cell'],
            cell_params=cell_params,
            dp_input_keep_prob=dp_input_keep_prob,
            dp_output_keep_prob=dp_output_keep_prob,
            # residual connections are added a little differently for GNMT
            residual_connections=False if self.params['attention_type'].startswith('gnmt')
                                 else residual_connections,
        ) for _ in range(self.params['decoder_layers'] - 1)
    ]

    last_cell_params = copy.deepcopy(cell_params)
    if self._weight_tied:
      last_cell_params['num_units'] = self._tgt_emb_size

    last_cell = single_cell(
            cell_class=self.params['core_cell'],
            cell_params=last_cell_params,
            dp_input_keep_prob=dp_input_keep_prob,
            dp_output_keep_prob=dp_output_keep_prob,
            # residual connections are added a little differently for GNMT
            residual_connections=False if self.params['attention_type'].startswith('gnmt')
                                 else residual_connections,
        )
    self._decoder_cells.append(last_cell)

    # pylint: disable=no-member
    tiled_enc_outputs = tf.contrib.seq2seq.tile_batch(
        encoder_outputs,
        multiplier=self._beam_width,
    )
    # pylint: disable=no-member
    tiled_enc_src_lengths = tf.contrib.seq2seq.tile_batch(
        enc_src_lengths,
        multiplier=self._beam_width,
    )
    attention_mechanism = self._build_attention(
        tiled_enc_outputs,
        tiled_enc_src_lengths,
    )

    if self.params['attention_type'].startswith('gnmt'):
      attention_cell = self._decoder_cells.pop(0)
      attention_cell = AttentionWrapper(
          attention_cell,
          attention_mechanism=attention_mechanism,
          attention_layer_size=None,  # don't use attention layer.
          output_attention=False,
          name="gnmt_attention",
      )
      attentive_decoder_cell = GNMTAttentionMultiCell(
          attention_cell,
          self._add_residual_wrapper(self._decoder_cells) if residual_connections else self._decoder_cells,
          use_new_attention=(self.params['attention_type'] == 'gnmt_v2')
      )
    else:  # non-GNMT
      attentive_decoder_cell = AttentionWrapper(
          # pylint: disable=no-member
          cell=tf.contrib.rnn.MultiRNNCell(self._decoder_cells),
          attention_mechanism=attention_mechanism,
      )
    batch_size_tensor = tf.constant(self._batch_size)
    embedding_fn = lambda ids: tf.cast(
        tf.nn.embedding_lookup(self._dec_emb_w, ids),
        dtype=self.params['dtype'],
    )
    decoder = BeamSearchDecoder(
        cell=attentive_decoder_cell,
        embedding=embedding_fn,
        start_tokens=tf.tile([self.GO_SYMBOL], [self._batch_size]),
        end_token=self.END_SYMBOL,
        initial_state=attentive_decoder_cell.zero_state(
            dtype=encoder_outputs.dtype,
            batch_size=batch_size_tensor * self._beam_width,
        ),
        beam_width=self._beam_width,
        output_layer=self._output_projection_layer,
        length_penalty_weight=self._length_penalty_weight
    )

    time_major = self.params.get("time_major", False)
    use_swap_memory = self.params.get("use_swap_memory", False)
    final_outputs, final_state, final_sequence_lengths = \
        tf.contrib.seq2seq.dynamic_decode(  # pylint: disable=no-member
            decoder=decoder,
            maximum_iterations=tf.reduce_max(enc_src_lengths) * 2,
            swap_memory=use_swap_memory,
            output_time_major=time_major,
        )

    return {'logits': final_outputs.predicted_ids[:, :, 0] if not time_major else
            tf.transpose(final_outputs.predicted_ids[:, :, 0], perm=[1, 0, 2]),
            'outputs': [final_outputs.predicted_ids[:, :, 0]],
            'final_state': final_state,
            'final_sequence_lengths': final_sequence_lengths}
