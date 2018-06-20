# Copyright (c) 2018 NVIDIA Corporation
"""
Conv-based encoder
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
from .encoder import Encoder

from open_seq2seq.parts.convs2s import embedding_layer
from open_seq2seq.parts.convs2s.utils import get_padding_bias, get_padding
from open_seq2seq.parts.convs2s import ffn_wn_layer, conv_wn_layer

# Default value used if max_input_length is not given
MAX_INPUT_LENGTH = 100


class ConvS2SEncoder(Encoder):
  """
  Fully convolutional Encoder of ConvS2S
  """
  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
      "encoder_layers": int,

      "src_emb_size": int,
      "src_vocab_size": int,
      "pad_embeddings_2_eight": bool,

      "embedding_dropout_keep_prob": float,
      "hidden_dropout_keep_prob": float,
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
      "conv_knum": list,
      "conv_kwidth": list,

      "att_layer_num": int,

      'max_input_length': int,
      'PAD_SYMBOL': int,

      'mask_paddings': bool
    })

  def __init__(self, params, model, name="convs2s_encoder_with_emb", mode='train'):
    """
    Initializes convolutional encoder with embeddings
    :param params: dictionary with encoder parameters
    Must define:
      * src_vocab_size - data vocabulary size
      * src_emb_size - size of embedding to use
      * mode - train or infer
    """
    super(ConvS2SEncoder, self).__init__(params, model, name=name, mode=mode)

    self._src_vocab_size = self.params['src_vocab_size']
    self._src_emb_size = self.params['src_emb_size']
    self.layers = []
    self._mode = mode


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

    inputs = input_dict['source_tensors'][0]
    source_length = input_dict['source_tensors'][1]

    with tf.variable_scope("encode"):
      # prepare encoder graph
      if len(self.layers) == 0:
        knum_list = self.params.get("conv_knum")
        kwidth_list = self.params.get("conv_kwidth")

        with tf.variable_scope("embedding"):
          self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
            self._src_vocab_size, self.params["src_emb_size"],
            pad2eight=self.params.get('pad_embeddings_2_eight', False), init_var=0.1)

        with tf.variable_scope("pos_embedding"):
          self.position_embedding_layer = embedding_layer.EmbeddingSharedWeights(
            self.params.get("max_input_length", MAX_INPUT_LENGTH), self._src_emb_size,
            pad2eight=self.params.get('pad_embeddings_2_eight', False), init_var=0.1)

        # linear projection before cnn layers
        self.layers.append(ffn_wn_layer.FeedFowardNetworkNormalized(self._src_emb_size, knum_list[0],
                                                                    dropout=self.params["embedding_dropout_keep_prob"],
                                                                    var_scope_name="linear_mapping_before_cnn_layers"))

        for i in range(self.params['encoder_layers']):
          in_dim = knum_list[i] if i == 0 else knum_list[i - 1]
          out_dim = knum_list[i]

          # linear projection is needed for residual connections if input and output of a cnn layer do not match
          if in_dim != out_dim:
            linear_proj = ffn_wn_layer.FeedFowardNetworkNormalized(in_dim, out_dim,
                                                             var_scope_name="linear_mapping_cnn_" + str(i+1),
                                                             dropout=1.0)
          else:
            linear_proj = None

          conv_layer = conv_wn_layer.Conv1DNetworkNormalized(in_dim, out_dim, kernel_width=kwidth_list[i],
                                                              mode=self.mode, layer_id=i+1,
                                                              hidden_dropout=self.params["hidden_dropout_keep_prob"],
                                                              conv_padding="SAME",
                                                              decode_padding=False)

          self.layers.append([linear_proj, conv_layer])

        # linear projection after cnn layers
        self.layers.append(ffn_wn_layer.FeedFowardNetworkNormalized(knum_list[-1], self._src_emb_size,
                                                                    dropout=1.0,
                                                                    var_scope_name="linear_mapping_after_cnn_layers"))

      inputs_attention_bias = get_padding_bias(inputs)
      encoder_inputs = self.embedding_softmax_layer(inputs)

      with tf.name_scope("add_pos_encoding"):
        pos_input = tf.range(0, tf.shape(encoder_inputs)[1], delta=1, dtype=tf.int32, name='range')
        pos_encoding = self.position_embedding_layer(pos_input)
        encoder_inputs = encoder_inputs + tf.cast(x=pos_encoding, dtype=encoder_inputs.dtype)

      if self.mode == "train":
        encoder_inputs = tf.nn.dropout(encoder_inputs, self.params["embedding_dropout_keep_prob"])

      # mask the paddings in the input given to cnn layers
      inputs_padding = get_padding(inputs, self.params.get('PAD_SYMBOL', 0))
      padding_mask = tf.cast(tf.expand_dims(tf.logical_not(inputs_padding), 2), encoder_inputs.dtype)
      encoder_inputs = encoder_inputs * padding_mask

      outputs, outputs_b, final_state = self._call(encoder_inputs)

    return {'outputs': outputs,
            'outputs_b': outputs_b,
            'inputs_attention_bias_cs2s': inputs_attention_bias,
            'state': final_state,
            'src_lengths': source_length, # should it include paddings or not?
            'embedding_softmax_layer': self.embedding_softmax_layer,
            #'position_embedding_layer': self.position_embedding_layer, # Should we share position embedding?
            'encoder_input': inputs}

  def _call(self, encoder_inputs):
    # Run inputs through the sublayers.
    with tf.variable_scope("linear_layer_before_cnn_layers"):
      outputs = self.layers[0](encoder_inputs)

    for i in range(1, len(self.layers) - 1):
      #if padding_mask is not None:
      #  outputs = outputs * padding_mask
      linear_proj, conv_layer = self.layers[i]

      with tf.variable_scope("layer_%d" % i):
        if linear_proj is not None:
          res_inputs = linear_proj(outputs)
        else:
          res_inputs = outputs
        outputs = conv_layer(outputs)
        outputs = (outputs + res_inputs) * tf.sqrt(0.5)

    with tf.variable_scope("linear_layer_after_cnn_layers"):
      outputs = self.layers[-1](outputs)

      #if padding_mask is not None:
      #  outputs = outputs * padding_mask
      # Gradients are scaled as the gradients from all decoder attention layers enters the encoder
      scale = 1.0 / (2.0 * self.params.get("att_layer_num", self.params["encoder_layers"]))
      outputs = (1.0 - scale) * tf.stop_gradient(outputs) + scale * outputs

      outputs_b = (outputs + encoder_inputs) * tf.sqrt(0.5)

      # Average of the encoder outputs is calculated as the final state of the encoder
      # it can be used for decoders which just accept the final state
      final_state = tf.reduce_mean(outputs_b, 1)
    return outputs, outputs_b, final_state


  @property
  def src_vocab_size(self):
    return self._src_vocab_size

  @property
  def src_emb_size(self):
    return self._src_emb_size

