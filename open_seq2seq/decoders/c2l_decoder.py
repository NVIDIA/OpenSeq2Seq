# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from .decoder import Decoder
from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv
from open_seq2seq.parts.cnns.attention_wrapper import BahdanauAttention, \
    LuongAttention, \
    _compute_attention


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
    training = (self.mode == "train")
    dropout_keep_prob = self.dropout_keep_prob if training else 1.0
    for layer in self.dense_layers:
      inputs = layer(inputs)
      inputs = tf.nn.dropout(x=inputs, keep_prob=dropout_keep_prob)
    return inputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    return tf.TensorShape([input_shape[0], self.output_dim])


class Conv2LetterDecoder(Decoder):
  """Convolution based attention decoder.
  """
  @staticmethod
  def get_required_params():
    return dict(Decoder.get_required_params(), **{
        'GO_SYMBOL': int,  # symbol id
        'END_SYMBOL': int,  # symbol id
        'tgt_vocab_size': int,
        'tgt_emb_size': int,
        'attention_params': dict,
        'convnet_params': dict,
        'fc_params': list,
    })

  @staticmethod
  def get_optional_params():
    return dict(Decoder.get_optional_params(), **{
        'dropout_keep_prob': float,
        'pos_embedding': bool,
    })

  def __init__(self, params, model, name='c2l_decoder', mode='train'):
    """Initializes CNN decoder with embedding.

    See parent class for arguments description.

    Config parameters:
    """
    super(Conv2LetterDecoder, self).__init__(params, model, name, mode)
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

    convnet_params = self.params['convnet_params']
    attention_params = self.params['attention_params']
    fc_params = self.params['fc_params']
    regularizer = self.params.get('regularizer', None)
    training = (self._mode == "train")
    dropout_keep_prob = self.params.get(
        'dropout_keep_prob', 1.0) if training else 1.0

    self._target_emb_layer = tf.get_variable(
        name='TargetEmbeddingMatrix',
        shape=[self._tgt_vocab_size, self._tgt_emb_size],
        dtype=tf.float32,
    )

    if self.params['pos_embedding']:
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
      decoder_output_positions = tf.range(
          0,
          tf.shape(tgt_inputs)[1],
          delta=1,
          dtype=tf.int32,
          name='positional_inputs'
      )

    output_projection_layer = FullyConnected(
        [hdim for hdim in fc_params] + [self._tgt_vocab_size],
        dropout_keep_prob=dropout_keep_prob,
        mode=self._mode,
    )

    normalization = convnet_params["normalization"]
    activation_fn = convnet_params["activation_fn"]
    convnet_layers = convnet_params["convnet_layers"]
    data_format = convnet_params["data_format"]
    normalization_params = {}
    if normalization is None:
      conv_block = conv_actv
    elif normalization == "batch_norm":
      conv_block = conv_bn_actv
      normalization_params['bn_momentum'] = self.params.get(
          'bn_momentum', 0.90)
      normalization_params['bn_epsilon'] = self.params.get('bn_epsilon', 1e-3)
    else:
      raise ValueError("Incorrect normalization")

    attention_dim = attention_params["attention_dim"]
    attention_type = attention_params["attention_type"]
    #num_heads = attention_params["num_heads"]

    attention_params_dict = {}
    if attention_type == "bahadanu":
      AttentionMechanism = BahdanauAttention
      attention_params_dict["normalize"] = False
    elif attention_type == "luong":
      AttentionMechanism = LuongAttention
      attention_params_dict["scale"] = False
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

    if self._mode == "train":
      tgt_input_vectors = tf.nn.embedding_lookup(
          self._target_emb_layer, tgt_inputs)
      if self.params['pos_embedding']:
        tgt_input_vectors += tf.nn.embedding_lookup(self.dec_pos_emb_layer,
                                                    decoder_output_positions)
      tgt_input_vectors = tf.cast(
          tgt_input_vectors,
          dtype=self.params['dtype'],
      )
    '''elif self._mode == "infer" or self._mode == "eval":
      embedding_fn = lambda ids: tf.cast(
          tf.nn.embedding_lookup(self._target_emb_layer, ids),
          dtype=self.params['dtype'],
      )'''

    '''if self._mode != "infer":
      maximum_iterations = tf.reduce_max(tgt_lengths)
    else:
      maximum_iterations = tf.reduce_max(enc_src_lengths)'''

    if self._mode == "train":
      conv_feats = tgt_input_vectors
      for idx_convnet in range(len(convnet_layers)):
        layer_type = convnet_layers[idx_convnet]['type']
        layer_repeat = convnet_layers[idx_convnet]['repeat']
        ch_out = convnet_layers[idx_convnet]['num_channels']
        kernel_size = convnet_layers[idx_convnet]['kernel_size']
        strides = convnet_layers[idx_convnet]['stride']
        padding = convnet_layers[idx_convnet]['padding']
        dropout_keep = convnet_layers[idx_convnet].get(
            'dropout_keep_prob', dropout_keep_prob) if training else 1.0
        
        for idx_layer in range(layer_repeat):
          conv_feats = conv_block(
                layer_type=layer_type,
                name="conv{}{}".format(
                    idx_convnet + 1, idx_layer + 1),
                inputs=conv_feats,
                filters=ch_out,
                kernel_size=kernel_size,
                activation_fn=activation_fn,
                strides=strides,
                padding=padding,
                regularizer=regularizer,
                training=training,
                data_format=data_format,
                use_residual=False,
                **normalization_params
          )
          #conv_feats = tf.nn.dropout(x=conv_feats, keep_prob=dropout_keep)
          attn_output, alignments, next_state = _compute_attention(attention_mechanism, conv_feats, None, None)
          conv_feats = conv_feats + attn_output
          print(attn_output)
          print(conv_feats)

      logits = output_projection_layer(tf.concat([conv_feats, attn_output], -1))      
      outputs = tf.argmax(logits, axis=-1)
      final_sequence_lengths = tgt_lengths
      print(alignments)

    return {
        'outputs': [outputs, alignments, enc_src_lengths, tgt_lengths],
        'logits': logits,
        'tgt_length': final_sequence_lengths,
    }
