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
      name="fully_connected",
  ):
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

  def call(self, inputs):
    for layer in self.dense_layers:
      inputs = layer(inputs)
    return inputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    return tf.TensorShape([input_shape[0], self.output_dim])


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
    })

  def __init__(self, params, model, name='las_decoder', mode='train'):
    """Initializes RNN decoder with embedding.

    See parent class for arguments description.

    Config parameters:
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
    """
    encoder_outputs = input_dict['encoder_output']['outputs']
    enc_src_lengths = input_dict['encoder_output']['src_length']

    self._batch_size = int(encoder_outputs.get_shape()[0])

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

    self._target_emb_layer = tf.get_variable(
        name='TargetEmbeddingMatrix',
        shape=[self._tgt_vocab_size, self._tgt_emb_size],
        dtype=tf.float32,
    )

    if self.params['pos_embedding']:
      self._pos_emb_size = int(encoder_outputs.get_shape()[-1])
      self._pos_emb_layer = tf.get_variable(
          name='PositionEmbeddingMatrix',
          shape=[1024, self._pos_emb_size],
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
              self._pos_emb_layer, encoder_output_positions),
          dtype=encoder_outputs.dtype
      )
      encoder_outputs = encoder_outputs + encoder_position_embeddings

    output_projection_layer = FullyConnected(
        [self._tgt_vocab_size]
    )

    rnn_cell = cells_dict[layer_type]

    dropout = tf.nn.rnn_cell.DropoutWrapper

    multirnn_cell = tf.nn.rnn_cell.MultiRNNCell(
        [dropout(rnn_cell(hidden_dim),
                 output_keep_prob=dropout_keep_prob)
         for _ in range(num_layers)]
    )

    attention_dim = attention_params["attention_dim"]
    attention_type = attention_params["attention_type"]

    attention_params_dict = {}
    if attention_type == "bahadanu":
      AttentionMechanism = BahdanauAttention
      attention_params_dict["normalize"] = False,
    elif attention_type == "chorowski":
      AttentionMechanism = LocationSensitiveAttention
      attention_params_dict["use_coverage"] = attention_params["use_coverage"]
      attention_params_dict["location_attn_type"] = attention_type
    elif attention_type == "zhaopeng":
      AttentionMechanism = LocationSensitiveAttention
      attention_params_dict["use_coverage"] = attention_params["use_coverage"]
      attention_params_dict["query_dim"] = hidden_dim
      attention_params_dict["location_attn_type"] = attention_type

    attention_mechanism = AttentionMechanism(
        num_units=attention_dim,
        memory=encoder_outputs,
        memory_sequence_length=enc_src_lengths,
        probability_fn=tf.nn.softmax,
        dtype=tf.get_variable_scope().dtype,
        **attention_params_dict
    )

    multirnn_cell_with_attention = AttentionWrapper(
        cell=multirnn_cell,
        attention_mechanism=attention_mechanism,
        attention_layer_size=hidden_dim,
        output_attention=True,
        alignment_history=True,
    )

    if self._mode == "train":
      tgt_input_vectors = tf.cast(
          tf.nn.embedding_lookup(self._target_emb_layer, tgt_inputs),
          dtype=self.params['dtype'],
      )
      helper = tf.contrib.seq2seq.TrainingHelper(
          inputs=tgt_input_vectors,
          sequence_length=tgt_lengths,
      )
    elif self._mode == "infer" or self._mode == "eval":
      embedding_fn = lambda ids: tf.cast(
          tf.nn.embedding_lookup(self._target_emb_layer, ids),
          dtype=self.params['dtype'],
      )
      helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
          embedding=embedding_fn,
          start_tokens=tf.fill([self._batch_size], self.GO_SYMBOL),
          end_token=self.END_SYMBOL,
      )

    if self._mode != "infer":
      maximum_iterations = tf.reduce_max(tgt_lengths)
    else:
      maximum_iterations = tf.reduce_max(enc_src_lengths)

    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=multirnn_cell_with_attention,
        helper=helper,
        initial_state=multirnn_cell_with_attention.zero_state(
            batch_size=self._batch_size, dtype=encoder_outputs.dtype,
        ),
        output_layer=output_projection_layer,
    )

    final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        impute_finished=False,
        maximum_iterations=maximum_iterations,
    )

    outputs = tf.argmax(final_outputs.rnn_output, axis=-1)
    alignments = tf.transpose(
        final_state.alignment_history.stack(), [1, 0, 2]
    )
    '''alignments = tf.expand_dims(alignments, axis=-1)
    alignments = tf.expand_dims(alignments, axis=1)

    summary = tf.summary.image(
      name='alignments',
      tensor=alignments[0],
      max_outputs=1,
    )'''

    '''bs, ln = tf.shape(encoder_outputs)[0], tf.shape(encoder_outputs)[1]
    indices = tf.constant([[i, j] for i in tf.range(bs) for j in tf.range(ln)])
    values = tf.reshape(outputs, [-1])
    sparse_outputs = tf.SparseTensor(indices, values, [bs, ln])'''

    return {
        'outputs': [outputs, alignments, enc_src_lengths],
        'logits': final_outputs.rnn_output,
        'tgt_length': final_sequence_lengths,
    }
