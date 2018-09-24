# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from open_seq2seq.parts.rnns.attention_wrapper import BahdanauAttention, \
    LuongAttention, \
    LocationSensitiveAttention, \
    AttentionWrapper
from open_seq2seq.parts.rnns.rnn_beam_search_decoder import BeamSearchDecoder
from open_seq2seq.parts.rnns.utils import single_cell
from open_seq2seq.parts.rnns.helper import TrainingHelper, GreedyEmbeddingHelper
from .decoder import Decoder

cells_dict = {
    "lstm": tf.nn.rnn_cell.BasicLSTMCell,
    "gru": tf.nn.rnn_cell.GRUCell,
}


class FullyConnected(tf.layers.Layer):
  """Fully connected layer
  """

  def __init__(
      self,
      hidden_dims,
      dropout_keep_prob=1.0,
      mode='train',
      name="fully_connected",
  ):
    """See parent class for arguments description.

    Config parameters:

    * **hidden_dims** (list) --- list of integers describing the hidden dimensions of a fully connected layer.
    * **dropout_keep_prob** (float, optional) - dropout input keep probability.
    """
    super(FullyConnected, self).__init__(name=name)

    self.dense_layers = []
    i = -1
    for i in range(len(hidden_dims) - 1):
      self.dense_layers.append(tf.layers.Dense(
          name="{}_{}".format(name, i), units=hidden_dims[i], use_bias=True, activation=tf.nn.relu)
      )
    self.dense_layers.append(tf.layers.Dense(
        name="{}_{}".format(name, i + 1), units=hidden_dims[i + 1], use_bias=True)
    )
    self.output_dim = hidden_dims[i + 1]
    self.mode = mode
    self.dropout_keep_prob = dropout_keep_prob

  def call(self, inputs):
    """
    Args:
      inputs: Similar to tf.layers.Dense layer inputs. Internally calls a stack of dense layers.
    """
    training = (self.mode == "train")
    dropout_keep_prob = self.dropout_keep_prob if training else 1.0
    for layer in self.dense_layers:
      inputs = tf.nn.dropout(x=inputs, keep_prob=dropout_keep_prob)
      inputs = layer(inputs)
    return inputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    return tf.TensorShape([input_shape[0], self.output_dim])


class ListenAttendSpellDecoder(Decoder):
  """Listen Attend Spell like decoder with attention mechanism.
  """
  @staticmethod
  def get_required_params():
    return dict(Decoder.get_required_params(), **{
        'GO_SYMBOL': int,  # symbol id
        'END_SYMBOL': int,  # symbol id
        'tgt_vocab_size': int,
        'tgt_emb_size': int,
        'attention_params': dict,
        'rnn_type': None,
        'hidden_dim': int,
        'num_layers': int,
    })

  @staticmethod
  def get_optional_params():
    return dict(Decoder.get_optional_params(), **{
        'dropout_keep_prob': float,
        'pos_embedding': bool,
        'beam_width': int,
        'use_language_model': bool,
    })

  def __init__(self, params, model, name='las_decoder', mode='train'):
    """Initializes decoder with embedding.

    See parent class for arguments description.

    Config parameters:

    * **GO_SYMBOL** (int) --- GO symbol id, must be the same as used in
      data layer.
    * **END_SYMBOL** (int) --- END symbol id, must be the same as used in
      data layer.
    * **tgt_vocab_size** (int) --- vocabulary size of the targets to use for final softmax.
    * **tgt_emb_size** (int) --- embedding size to use.
    * **attention_params** (dict) - parameters for attention mechanism.
    * **rnn_type** (String) - String indicating the rnn type. Accepts ['lstm', 'gru'].
    * **hidden_dim** (int) - Hidden domension to be used the RNN decoder.
    * **num_layers** (int) - Number of decoder RNN layers.
    * **dropout_keep_prob** (float, optional) - dropout input keep probability.
    * **pos_embedding** (bool, optional) - Whether to use encoder and decoder positional embedding. Default is False.
    * **beam_width** (int, optional) - Beam width used while decoding with beam search. Uses greedy decoding if the value is set to 1. Default is 1.
    * **use_language_model** (bool, optional) - Boolean indicating whether to use language model for decoding. Default is False.
    """
    super(ListenAttendSpellDecoder, self).__init__(params, model, name, mode)
    self.GO_SYMBOL = self.params['GO_SYMBOL']
    self.END_SYMBOL = self.params['END_SYMBOL']
    self._tgt_vocab_size = self.params['tgt_vocab_size']
    self._tgt_emb_size = self.params['tgt_emb_size']

  def _decode(self, input_dict):
    """Decodes representation into data.

    Args:
      input_dict (dict): Python dictionary with inputs to decoder.


    Config parameters:

    * **src_inputs** --- Decoder input Tensor of shape [batch_size, time, dim]
      or [time, batch_size, dim].
    * **src_lengths** --- Decoder input lengths Tensor of shape [batch_size]
    * **tgt_inputs** --- Only during training. labels Tensor of the
      shape [batch_size, time] or [time, batch_size].
    * **tgt_lengths** --- Only during training. labels lengths
      Tensor of the shape [batch_size].

    Returns:
      dict: Python dictionary with:
      * outputs - [predictions, alignments, enc_src_lengths].
        predictions are the final predictions of the model. tensor of shape [batch_size, time].
        alignments are the attention probabilities if attention is used. None if 'plot_attention' in attention_params is set to False.
        enc_src_lengths are the lengths of the input. tensor of shape [batch_size].
      * logits - logits with the shape=[batch_size, output_dim].
      * tgt_length - tensor of shape [batch_size] indicating the predicted sequence lengths.
    """
    encoder_outputs = input_dict['encoder_output']['outputs']
    enc_src_lengths = input_dict['encoder_output']['src_length']

    self._batch_size = int(encoder_outputs.get_shape()[0])
    self._beam_width = self.params.get("beam_width", 1)

    tgt_inputs = None
    tgt_lengths = None
    if 'target_tensors' in input_dict:
      tgt_inputs = input_dict['target_tensors'][0]
      tgt_lengths = input_dict['target_tensors'][1]
      tgt_inputs = tf.concat(
          [tf.fill([self._batch_size, 1], self.GO_SYMBOL), tgt_inputs[:, :-1]], -1)

    layer_type = self.params['rnn_type']
    num_layers = self.params['num_layers']
    attention_params = self.params['attention_params']
    hidden_dim = self.params['hidden_dim']
    dropout_keep_prob = self.params.get(
        'dropout_keep_prob', 1.0) if self._mode == "train" else 1.0

    # To-Do Seperate encoder and decoder position embeddings
    use_positional_embedding = self.params.get("pos_embedding", False)
    use_language_model = self.params.get("use_language_model", False)
    use_beam_search_decoder = (
        self._beam_width != 1) and (self._mode == "infer")

    self._target_emb_layer = tf.get_variable(
        name='TargetEmbeddingMatrix',
        shape=[self._tgt_vocab_size, self._tgt_emb_size],
        dtype=tf.float32,
    )

    if use_positional_embedding:
      self.enc_pos_emb_size = int(encoder_outputs.get_shape()[-1])
      self.enc_pos_emb_layer = tf.get_variable(
          name='EncoderPositionEmbeddingMatrix',
          shape=[1024, self.enc_pos_emb_size],
          dtype=tf.float32,
      )
      encoder_output_positions = tf.range(
          0,
          tf.shape(encoder_outputs)[1],
          delta=1,
          dtype=tf.int32,
          name='positional_inputs'
      )
      encoder_position_embeddings = tf.cast(
          tf.nn.embedding_lookup(
              self.enc_pos_emb_layer, encoder_output_positions),
          dtype=encoder_outputs.dtype
      )
      encoder_outputs += encoder_position_embeddings

      self.dec_pos_emb_size = self._tgt_emb_size
      self.dec_pos_emb_layer = tf.get_variable(
          name='DecoderPositionEmbeddingMatrix',
          shape=[1024, self.dec_pos_emb_size],
          dtype=tf.float32,
      )

    output_projection_layer = FullyConnected(
        [self._tgt_vocab_size],
        dropout_keep_prob=dropout_keep_prob,
        mode=self._mode,
    )

    rnn_cell = cells_dict[layer_type]

    dropout = tf.nn.rnn_cell.DropoutWrapper

    multirnn_cell = tf.nn.rnn_cell.MultiRNNCell(
        [dropout(rnn_cell(hidden_dim),
                 output_keep_prob=dropout_keep_prob)
         for _ in range(num_layers)]
    )

    if use_beam_search_decoder:
      encoder_outputs = tf.contrib.seq2seq.tile_batch(
          encoder_outputs,
          multiplier=self._beam_width,
      )
      enc_src_lengths = tf.contrib.seq2seq.tile_batch(
          enc_src_lengths,
          multiplier=self._beam_width,
      )

    attention_dim = attention_params["attention_dim"]
    attention_type = attention_params["attention_type"]
    num_heads = attention_params["num_heads"]
    plot_attention = attention_params["plot_attention"]
    if plot_attention:
      if use_beam_search_decoder:
        plot_attention = False
        print("Plotting Attention is disabled for Beam Search Decoding")
      if num_heads != 1:
        plot_attention = False
        print("Plotting Attention is disabled for Multi Head Attention")
      if self.params['dtype'] != tf.float32:
        plot_attention = False
        print("Plotting Attention is disabled for Mixed Precision Mode")

    attention_params_dict = {}
    if attention_type == "bahadanu":
      AttentionMechanism = BahdanauAttention
      attention_params_dict["normalize"] = False,
    elif attention_type == "chorowski":
      AttentionMechanism = LocationSensitiveAttention
      attention_params_dict["use_coverage"] = attention_params["use_coverage"]
      attention_params_dict["location_attn_type"] = attention_type
      attention_params_dict["location_attention_params"] = {
          'filters': 10, 'kernel_size': 101}
    elif attention_type == "zhaopeng":
      AttentionMechanism = LocationSensitiveAttention
      attention_params_dict["use_coverage"] = attention_params["use_coverage"]
      attention_params_dict["query_dim"] = hidden_dim
      attention_params_dict["location_attn_type"] = attention_type

    attention_mechanism = []

    for head in range(num_heads):
      attention_mechanism.append(
          AttentionMechanism(
              num_units=attention_dim,
              memory=encoder_outputs,
              memory_sequence_length=enc_src_lengths,
              probability_fn=tf.nn.softmax,
              dtype=tf.get_variable_scope().dtype,
              **attention_params_dict
          )
      )

    multirnn_cell_with_attention = AttentionWrapper(
        cell=multirnn_cell,
        attention_mechanism=attention_mechanism,
        attention_layer_size=[hidden_dim for i in range(num_heads)],
        output_attention=True,
        alignment_history=plot_attention,
    )

    if self._mode == "train":
      decoder_output_positions = tf.range(
          0,
          tf.shape(tgt_inputs)[1],
          delta=1,
          dtype=tf.int32,
          name='positional_inputs'
      )
      tgt_input_vectors = tf.nn.embedding_lookup(
          self._target_emb_layer, tgt_inputs)
      if use_positional_embedding:
        tgt_input_vectors += tf.nn.embedding_lookup(self.dec_pos_emb_layer,
                                                    decoder_output_positions)
      tgt_input_vectors = tf.cast(
          tgt_input_vectors,
          dtype=self.params['dtype'],
      )
      # helper = tf.contrib.seq2seq.TrainingHelper(
      helper = TrainingHelper(
          inputs=tgt_input_vectors,
          sequence_length=tgt_lengths,
      )
    elif self._mode == "infer" or self._mode == "eval":
      embedding_fn = lambda ids: tf.cast(
          tf.nn.embedding_lookup(self._target_emb_layer, ids),
          dtype=self.params['dtype'],
      )
      pos_embedding_fn = None
      if use_positional_embedding:
        pos_embedding_fn = lambda ids: tf.cast(
            tf.nn.embedding_lookup(self.dec_pos_emb_layer, ids),
            dtype=self.params['dtype'],
        )

      # helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
      helper = GreedyEmbeddingHelper(
          embedding=embedding_fn,
          start_tokens=tf.fill([self._batch_size], self.GO_SYMBOL),
          end_token=self.END_SYMBOL,
          positional_embedding=pos_embedding_fn
      )

    if self._mode != "infer":
      maximum_iterations = tf.reduce_max(tgt_lengths)
    else:
      maximum_iterations = tf.reduce_max(enc_src_lengths)

    if not use_beam_search_decoder:
      decoder = tf.contrib.seq2seq.BasicDecoder(
          cell=multirnn_cell_with_attention,
          helper=helper,
          initial_state=multirnn_cell_with_attention.zero_state(
              batch_size=self._batch_size, dtype=encoder_outputs.dtype,
          ),
          output_layer=output_projection_layer,
      )
    else:
      batch_size_tensor = tf.constant(self._batch_size)
      decoder = BeamSearchDecoder(
          cell=multirnn_cell_with_attention,
          embedding=embedding_fn,
          start_tokens=tf.tile([self.GO_SYMBOL], [self._batch_size]),
          end_token=self.END_SYMBOL,
          initial_state=multirnn_cell_with_attention.zero_state(
              dtype=encoder_outputs.dtype,
              batch_size=batch_size_tensor * self._beam_width,
          ),
          beam_width=self._beam_width,
          output_layer=output_projection_layer,
          length_penalty_weight=0.0,
      )

    final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        impute_finished=self.mode != "infer",
        maximum_iterations=maximum_iterations,
    )

    if plot_attention:
      alignments = tf.transpose(
          final_state.alignment_history[0].stack(), [1, 0, 2]
      )
    else:
      alignments = None

    if not use_beam_search_decoder:
      outputs = tf.argmax(final_outputs.rnn_output, axis=-1)
      logits = final_outputs.rnn_output
      return_outputs = [outputs, alignments, enc_src_lengths]
    else:
      outputs = final_outputs.predicted_ids[:, :, 0]
      logits = final_outputs.predicted_ids[:, :, 0]
      return_outputs = [outputs, enc_src_lengths]

    if self.mode == "eval":
      max_len = tf.reduce_max(tgt_lengths)
      logits = tf.while_loop(
          lambda logits: max_len > tf.shape(logits)[1],
          lambda logits: tf.concat([logits, tf.fill(
              [tf.shape(logits)[0], 1, tf.shape(logits)[2]], tf.cast(1.0, self.params['dtype']))], 1),
          loop_vars=[logits],
          back_prop=False,
      )

    return {
        'outputs': return_outputs,
        'logits': logits,
        'tgt_length': final_sequence_lengths,
    }
