# Copyright (c) 2017 NVIDIA Corporation
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from .model_utils import create_rnn_cell, getdtype
from .model_base import ModelBase
from .gnmt import GNMTAttentionMultiCell, gnmt_residual_fn
import copy


class BasicSeq2SeqWithAttention(ModelBase):
  """
  Basic sequence-2-sequence model with attention
  """
  def _build_encoder(self, src_inputs, src_lengths=None):
    """
    Builds decoder graph
    :param src_inputs: sources, Tensor of shape [batch_size, max_num_steps]
    :param src_lengths: (Optional) source lengths, Tensor of shape [batch_size]
    :return: (encoder_outputs, encoder_state, src_lengths)
    """
    src_vocab_size = self.model_params['src_vocab_size']
    src_emb_size = self.model_params['src_emb_size']
    with tf.variable_scope("Encoder"):
      self._src_w = tf.get_variable(name="W_src_embedding",
                                    shape=[src_vocab_size, src_emb_size],
                                    dtype=getdtype())
      # embedded_inputs will be [batch_size, time, src_emb_size]
      embedded_inputs = tf.nn.embedding_lookup(self._src_w, src_inputs)

      if self._encoder_type == "unidirectional":
        encoder_outputs, encoder_state = self._unidirectional_encoder(
          embedded_inputs,
          src_lengths)
      elif self._encoder_type == "bidirectional":
        encoder_output, encoder_state = self._bidirectional_encoder(
            embedded_inputs,
            src_lengths)
        encoder_outputs = tf.concat(encoder_output, 2)
      elif self._encoder_type == "gnmt":
        encoder_outputs, encoder_state = self._gnmt_encoder(
          embedded_inputs,
          src_lengths)
      else:
        raise ValueError('Unknown encoder type')
    return encoder_outputs, encoder_state, src_lengths

  # Encoders
  def _unidirectional_encoder(self, embedded_inputs, src_lengths):
    """
    Creates graph for unidirectional encoder. TODO: add param tensor shapes
    :param embedded_inputs:
    :param src_lengths:
    :return:
    """
    cell_params = copy.deepcopy(self.model_params)
    cell_params["num_units"] = self.model_params['encoder_cell_units']
    encoder_cell_fw = create_rnn_cell(cell_type=self.model_params['encoder_cell_type'],
                                      cell_params=cell_params,
                                      num_layers=self.model_params['encoder_layers'],
                                      dp_input_keep_prob=self.model_params['encoder_dp_input_keep_prob'] if self._mode == "train" else 1.0,
                                      dp_output_keep_prob=self.model_params['encoder_dp_output_keep_prob'] if self._mode == "train" else 1.0,
                                      residual_connections=self.model_params['encoder_use_skip_connections'])
    return tf.nn.dynamic_rnn(
      cell = encoder_cell_fw,
      inputs = embedded_inputs,
      sequence_length = src_lengths,
      dtype = getdtype(), swap_memory= False if 'use_swap_memory' not in self.model_params else self.model_params['use_swap_memory'])

  def _bidirectional_encoder(self, embedded_inputs, src_lengths):
    """
    Creates graph for bi-directional encoder. TODO: add param tensor shapes
    :param embedded_inputs:
    :param src_lengths:
    :return:
    """
    cell_params = copy.deepcopy(self.model_params)
    cell_params["num_units"] = self.model_params['encoder_cell_units']
    encoder_cell_fw = create_rnn_cell(cell_type=self.model_params['encoder_cell_type'],
                                      cell_params=cell_params,
                                      num_layers=self.model_params['encoder_layers'],
                                      dp_input_keep_prob=self.model_params['encoder_dp_input_keep_prob'] if self._mode == "train" else 1.0,
                                      dp_output_keep_prob=self.model_params['encoder_dp_output_keep_prob'] if self._mode == "train" else 1.0,
                                      residual_connections=self.model_params['encoder_use_skip_connections'])

    encoder_cell_bw = create_rnn_cell(cell_type=self.model_params['encoder_cell_type'],
                                      cell_params=cell_params,
                                      num_layers=self.model_params['encoder_layers'],
                                      dp_input_keep_prob=self.model_params['encoder_dp_input_keep_prob'] if self._mode == "train" else 1.0,
                                      dp_output_keep_prob=self.model_params['encoder_dp_output_keep_prob'] if self._mode == "train" else 1.0,
                                      residual_connections=self.model_params['encoder_use_skip_connections'])
    return tf.nn.bidirectional_dynamic_rnn(
        cell_fw=encoder_cell_fw,
        cell_bw = encoder_cell_bw,
        inputs=embedded_inputs,
        sequence_length=src_lengths,
        dtype=getdtype(), swap_memory= False if 'use_swap_memory' not in self.model_params else self.model_params['use_swap_memory'])

  def _gnmt_encoder(self, embedded_inputs, src_lengths):
    if self.model_params['encoder_layers'] < 2:
      raise ValueError("GNMT encoder must have at least 2 layers")

    cell_params = copy.deepcopy(self.model_params)
    cell_params["num_units"] = self.model_params['encoder_cell_units']
    encoder_l1_cell_fw = create_rnn_cell(cell_type=self.model_params['encoder_cell_type'],
                                         cell_params=cell_params,
                                         num_layers=1,
                                         dp_input_keep_prob=1.0,
                                         dp_output_keep_prob=1.0,
                                         residual_connections=False)

    encoder_l1_cell_bw = create_rnn_cell(cell_type=self.model_params['encoder_cell_type'],
                                         cell_params=cell_params,
                                         num_layers=1,
                                         dp_input_keep_prob=1.0,
                                         dp_output_keep_prob=1.0,
                                         residual_connections=False)

    _encoder_output, _ = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=encoder_l1_cell_fw,
      cell_bw=encoder_l1_cell_bw,
      inputs=embedded_inputs,
      sequence_length=src_lengths,
      dtype=getdtype(),
      swap_memory=False if 'use_swap_memory' not in self.model_params else self.model_params['use_swap_memory'])

    encoder_l1_outputs = tf.concat(_encoder_output, 2)

    encoder_cells = create_rnn_cell(cell_type=self.model_params['encoder_cell_type'],
                                    cell_params=cell_params,
                                    num_layers=self.model_params['encoder_layers'] - 1,
                                    dp_input_keep_prob=self.model_params['encoder_dp_input_keep_prob'] if self._mode == "train" else 1.0,
                                    dp_output_keep_prob=self.model_params['encoder_dp_output_keep_prob'] if self._mode == "train" else 1.0,
                                    residual_connections=False,
                                    wrap_to_multi_rnn=False)
    # add residual connections starting from the third layer
    for idx, cell in enumerate(encoder_cells):
      if idx>0:
        encoder_cells[idx] = tf.contrib.rnn.ResidualWrapper(cell)

    return tf.nn.dynamic_rnn(
      cell=tf.contrib.rnn.MultiRNNCell(encoder_cells),
      inputs=encoder_l1_outputs,
      sequence_length=src_lengths,
      dtype=getdtype(),
      swap_memory=False if 'use_swap_memory' not in self.model_params else self.model_params['use_swap_memory'])


  def _build_attention(self, encoder_outputs, encoder_sequence_length):
    """
    Builds Attention part of the graph.
    Currently supports "bahdanau" and "luong"
    :param encoder_outputs:
    :param encoder_sequence_length:
    :return:
    """
    with tf.variable_scope("Attention"):
      attention_depth = self.model_params['attention_layer_size']
      if self.model_params['attention_type'] == 'bahdanau':
        bah_normalize = self.model_params['bahdanau_normalize'] if 'bahdanau_normalize' in self.model_params else False
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=attention_depth,
                                             memory=encoder_outputs, normalize = bah_normalize,
                                             memory_sequence_length=encoder_sequence_length,
                                             probability_fn=tf.nn.softmax)
      elif self.model_params['attention_type'] == 'luong':
        luong_scale = self.model_params['luong_scale'] if 'luong_scale' in self.model_params else False
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=attention_depth,
                                             memory=encoder_outputs, scale = luong_scale,
                                             memory_sequence_length=encoder_sequence_length,
                                             probability_fn=tf.nn.softmax)
      elif self.model_params['attention_type'] == 'gnmt' or self.model_params['attention_type'] == 'gnmt_v2':
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=attention_depth,
                                                                   memory=encoder_outputs, normalize=True,
                                                                   memory_sequence_length=encoder_sequence_length,
                                                                   probability_fn=tf.nn.softmax)
      else:
        raise ValueError('Unknown Attention Type')

      return attention_mechanism

  def _build_decoder(self,
                     encoder_outputs,
                     enc_src_lengths,
                     tgt_inputs = None,
                     tgt_lengths = None,
                     GO_SYMBOL = 1,
                     END_SYMBOL = 2,
                     out_layer_activation = None):
    """
    Builds decoder part of the graph, for training and inference
    TODO: add param tensor shapes
    :param encoder_outputs:
    :param enc_src_lengths:
    :param tgt_inputs:
    :param tgt_lengths:
    :param GO_SYMBOL:
    :param END_SYMBOL:
    :param out_layer_activation:
    :return:
    """
    def _add_residual_wrapper(cells, start_ind=1):
      for idx, cell in enumerate(cells):
        if idx>=start_ind:
          cells[idx] = tf.contrib.rnn.ResidualWrapper(cell, residual_fn=gnmt_residual_fn)
      return  cells

    with tf.variable_scope("Decoder"):
      tgt_vocab_size = self.model_params['tgt_vocab_size']
      tgt_emb_size = self.model_params['tgt_emb_size']
      self._tgt_w = tf.get_variable(name='W_tgt_embedding',
                                    shape=[tgt_vocab_size, tgt_emb_size], dtype=getdtype())
      batch_size = self.model_params['batch_size']

      cell_params = copy.deepcopy(self.model_params)
      cell_params["num_units"] = self.model_params['decoder_cell_units']
      decoder_cells = create_rnn_cell(cell_type=self.model_params['decoder_cell_type'],
                                      cell_params=cell_params,
                                      num_layers=self.model_params['decoder_layers'],
                                      dp_input_keep_prob=self.model_params[
                                       'decoder_dp_input_keep_prob'] if self._mode == "train" else 1.0,
                                      dp_output_keep_prob=self.model_params[
                                       'decoder_dp_output_keep_prob'] if self._mode == "train" else 1.0,
                                      residual_connections=False if self.model_params['attention_type'].startswith('gnmt')
                                      else self.model_params['decoder_use_skip_connections'],
                                      wrap_to_multi_rnn=not self.model_params['attention_type'].startswith('gnmt'))

      output_layer = layers_core.Dense(tgt_vocab_size, use_bias=False,
                                       activation = out_layer_activation)

      if self.mode == "infer":
        if self._decoder_type == "beam_search":
          self._length_penalty_weight = 0.0 if "length_penalty" not in self.model_params else self.model_params[
            "length_penalty"]
          # beam_width of 1 should be same as argmax decoder
          self._beam_width = 1 if "beam_width" not in self.model_params else self.model_params["beam_width"]
          tiled_enc_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self._beam_width)
          tiled_enc_src_lengths = tf.contrib.seq2seq.tile_batch(enc_src_lengths, multiplier=self._beam_width)
          attention_mechanism = self._build_attention(tiled_enc_outputs, tiled_enc_src_lengths)

          if self.model_params['attention_type'].startswith('gnmt'):
            attention_cell = decoder_cells.pop(0)
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
              attention_cell,
              attention_mechanism = attention_mechanism,
              attention_layer_size=None,  # don't use attenton layer.
              output_attention=False,
              name="gnmt_attention")
            attentive_decoder_cell = GNMTAttentionMultiCell(
              attention_cell, _add_residual_wrapper(decoder_cells),
              use_new_attention=(self.model_params['attention_type']=='gnmt_v2'))
          else:
            attentive_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cells,
                                                                         attention_mechanism=attention_mechanism)
          batch_size_tensor = tf.constant(batch_size)
          decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=attentive_decoder_cell,
            embedding=self._tgt_w,
            start_tokens=tf.tile([GO_SYMBOL], [batch_size]),
            end_token=END_SYMBOL,
            initial_state=attentive_decoder_cell.zero_state(dtype=getdtype(),
                                                            batch_size=batch_size_tensor * self._beam_width),
            beam_width=self._beam_width,
            output_layer=output_layer,
            length_penalty_weight=self._length_penalty_weight)
        else:
          attention_mechanism = self._build_attention(encoder_outputs, enc_src_lengths)
          if self.model_params['attention_type'].startswith('gnmt'):
            attention_cell = decoder_cells.pop(0)
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
              attention_cell,
              attention_mechanism = attention_mechanism,
              attention_layer_size=None,
              output_attention=False,
              name="gnmt_attention")
            attentive_decoder_cell = GNMTAttentionMultiCell(
              attention_cell, _add_residual_wrapper(decoder_cells),
              use_new_attention=(self.model_params['attention_type']=='gnmt_v2'))
          else:
            attentive_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cells,
                                                                         attention_mechanism=attention_mechanism)
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=self._tgt_w,
            start_tokens=tf.fill([batch_size], GO_SYMBOL),
            end_token=END_SYMBOL)
          decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=attentive_decoder_cell,
            helper=helper,
            initial_state=attentive_decoder_cell.zero_state(batch_size=batch_size, dtype=getdtype()),
            output_layer=output_layer)

      elif self.mode == "train":
        attention_mechanism = self._build_attention(encoder_outputs, enc_src_lengths)
        if self.model_params['attention_type'].startswith('gnmt'):
          attention_cell = decoder_cells.pop(0)
          attention_cell = tf.contrib.seq2seq.AttentionWrapper(
            attention_cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=None,
            output_attention=False,
            name="gnmt_attention")
          attentive_decoder_cell = GNMTAttentionMultiCell(
            attention_cell, _add_residual_wrapper(decoder_cells),
            use_new_attention=(self.model_params['attention_type'] == 'gnmt_v2'))
        else:
          attentive_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cells,
                                                                       attention_mechanism=attention_mechanism)
        input_vectors = tf.nn.embedding_lookup(self._tgt_w, tgt_inputs)
        helper = tf.contrib.seq2seq.TrainingHelper(
          inputs = input_vectors,
          sequence_length = tgt_lengths)

        decoder = tf.contrib.seq2seq.BasicDecoder(
          cell=attentive_decoder_cell,
          helper=helper,
          output_layer=output_layer,
          initial_state=attentive_decoder_cell.zero_state(batch_size, dtype=getdtype()))
      else:
        raise NotImplementedError("Unknown mode")

      final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
        decoder = decoder,
        impute_finished=False if self._decoder_type == "beam_search" else True,
        maximum_iterations=tf.reduce_max(tgt_lengths) if self._mode == 'train' else tf.reduce_max(enc_src_lengths)*2,
        swap_memory = False if 'use_swap_memory' not in self.model_params else self.model_params['use_swap_memory'])

      return final_outputs, final_state, final_sequence_lengths

  def _build_forward_pass_graph(self,
                                source_sequence,
                                src_length=None,
                                target_sequence=None,
                                tgt_length=None,
                                gpu_id=0):
    """
    Builds graph representing forward pass of the model
    :param source_sequence: [batch_size, T]
    :param src_length: [batch_size]
    :param target_sequence: [batch_size, T]
    :param tgt_length: [batch_size]
    :return:
    """
    self._encoder_type = "unidirectional" if "encoder_type" not in self.model_params else self.model_params["encoder_type"]
    self._use_attention = False if "use_attention" not in self.model_params else self.model_params["use_attention"]
    self._decoder_type = "greedy" if "decoder_type" not in self.model_params else self.model_params["decoder_type"]
    self._temp = self.model_params["softmax_temperature"] if "softmax_temperature" in self.model_params else None

    encoder_outputs, _, enc_src_lengths = self._build_encoder(src_inputs = source_sequence,
                                                              src_lengths = src_length)
    def temp(temp_input):
      t = self._temp if self._temp is not None else 0.5
      return tf.scalar_mul(1.0/t, temp_input)

    final_outputs, final_state, final_sequence_lengths = self._build_decoder(
      encoder_outputs = encoder_outputs,
      enc_src_lengths = enc_src_lengths,
      tgt_inputs = target_sequence,
      tgt_lengths = None if tgt_length is None else tgt_length,
      GO_SYMBOL=1,
      END_SYMBOL=2,
      out_layer_activation=temp if self._temp is not None else None)

    if gpu_id == 0:
      self._final_outputs = final_outputs

    if target_sequence is not None and tgt_length is not None and self._decoder_type != "beam_search":
      current_ts = tf.to_int32(tf.minimum(tf.shape(target_sequence)[1], tf.shape(final_outputs.rnn_output)[1])) - 1
      target_sequence = tf.slice(target_sequence,
                                 begin=[0, 1],
                                 size=[-1, current_ts])
      mask_ = tf.sequence_mask(lengths=tgt_length - 1,
                               maxlen=current_ts,
                               dtype=tf.float32)
      logits = tf.slice(final_outputs.rnn_output, begin=[0, 0, 0], size=[-1, current_ts, -1])
      average_across_timestep = False if "average_across_timesteps" not in self.model_params else self.model_params["average_across_timesteps"]
      if average_across_timestep:
        loss = tf.contrib.seq2seq.sequence_loss(logits = logits,
                                                targets = target_sequence,
                                                weights = mask_,
                                                average_across_timesteps=True,
                                                average_across_batch=True,
                                                softmax_loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits)
      else:
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tf.reshape(target_sequence, shape=[-1]),
          logits=tf.reshape(logits, shape=[-1, self.model_params['tgt_vocab_size']]))
        loss = (tf.reduce_sum(crossent * tf.reshape(mask_, shape=[-1])) / self.per_gpu_batch_size)
      return final_outputs, loss
    else:
      print("Inference Mode. Loss part of graph isn't built.")

  @property
  def final_outputs(self):
    return self._final_outputs
