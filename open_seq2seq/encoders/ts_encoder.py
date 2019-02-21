# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from .encoder import Encoder
from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv, \
  conv_ln_actv, conv_in_actv, conv_bn_res_bn_actv
from open_seq2seq.parts.transformer import attention_layer, ffn_layer, utils
from open_seq2seq.parts.transformer.common import PrePostProcessingWrapper, \
                                    LayerNormalization, Transformer_BatchNorm

class TSSEncoder(Encoder):
  """General time delay neural network (TDNN) encoder. Fully convolutional model
  """

  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
        'dropout_keep_prob': float,
        'convnet_layers': list,
        'activation_fn': None,  # any valid callable
        "encoder_layers": int,
        "hidden_size": int,
        "num_heads": int,
        "attention_dropout": float,
        "filter_size": int,
        "relu_dropout": float,
        "layer_postprocess_dropout": float,
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
        'data_format': ['channels_first', 'channels_last'],
        'normalization': [None, 'batch_norm', 'layer_norm', 'instance_norm'],
        'bn_momentum': float,
        'bn_epsilon': float,
    })

  def __init__(self, params, model, name="transformer_speech_encoder",
               mode='train'):
    """TSSEncoder encoder constructor.

    See parent class for arguments description.

    Config parameters:

    * **dropout_keep_prop** (float) --- keep probability for dropout.
    * **convnet_layers** (list) --- list with the description of convolutional
      layers. For example::
        "convnet_layers": [
          {
            "type": "conv1d", "repeat" : 5,
            "kernel_size": [7], "stride": [1],
            "num_channels": 250, "padding": "SAME"
          },
          {
            "type": "conv1d", "repeat" : 3,
            "kernel_size": [11], "stride": [1],
            "num_channels": 500, "padding": "SAME"
          },
          {
            "type": "conv1d", "repeat" : 1,
            "kernel_size": [32], "stride": [1],
            "num_channels": 1000, "padding": "SAME"
          },
          {
            "type": "conv1d", "repeat" : 1,
            "kernel_size": [1], "stride": [1],
            "num_channels": 1000, "padding": "SAME"
          },
        ]
    * **activation_fn** --- activation function to use.
    * **data_format** (string) --- could be either "channels_first" or
      "channels_last". Defaults to "channels_last".
    * **normalization** --- normalization to use. Accepts [None, 'batch_norm'].
      Use None if you don't want to use normalization. Defaults to 'batch_norm'.
    * **bn_momentum** (float) --- momentum for batch norm. Defaults to 0.90.
    * **bn_epsilon** (float) --- epsilon for batch norm. Defaults to 1e-3.
    """
    super(TSSEncoder, self).__init__(params, model, name, mode)
    self.layers = []
    self.output_normalization = None
    self._mode = mode

    self.norm_params = self.params.get("norm_params", {"type": "layernorm_L2"})
    self.regularizer = self.params.get("regularizer", None)
    if self.regularizer != None:
      self.regularizer_params = params.get("regularizer_params", {'scale': 0.0})
      self.regularizer=self.regularizer(self.regularizer_params['scale']) \
        if self.regularizer_params['scale'] > 0.0 else None


  def _call(self, encoder_inputs):
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.variable_scope("layer_%d" % n):
        with tf.variable_scope("self_attention"):
          encoder_inputs = self_attention_layer(encoder_inputs, None)
        with tf.variable_scope("ffn"):
          encoder_inputs = feed_forward_network(encoder_inputs, None)

    return self.output_normalization(encoder_inputs)

  def _encode(self, input_dict):
    """Creates TensorFlow graph for Wav2Letter like encoder.

    Args:
      input_dict (dict): input dictionary that has to contain
          the following fields::
            input_dict = {
              "source_tensors": [
                src_sequence (shape=[batch_size, sequence length, num features]),
                src_length (shape=[batch_size])
              ]
            }

    Returns:
      dict: dictionary with the following tensors::

        {
          'outputs': hidden state, shape=[batch_size, sequence length, n_hidden]
          'src_length': tensor, shape=[batch_size]
        }
    """

    source_sequence, src_length = input_dict['source_tensors']

    training = (self._mode == "train")
    dropout_keep_prob = self.params['dropout_keep_prob'] if training else 1.0
    regularizer = self.params.get('regularizer', None)
    data_format = self.params.get('data_format', 'channels_last')
    normalization = self.params.get('normalization', 'batch_norm')

    normalization_params = {}
    if normalization is None:
      conv_block = conv_actv
    elif normalization == "batch_norm":
      conv_block = conv_bn_actv
      normalization_params['bn_momentum'] = self.params.get(
          'bn_momentum', 0.90)
      normalization_params['bn_epsilon'] = self.params.get('bn_epsilon', 1e-3)
    elif normalization == "layer_norm":
      conv_block = conv_ln_actv
    elif normalization == "instance_norm":
      conv_block = conv_in_actv
    else:
      raise ValueError("Incorrect normalization")

    conv_inputs = source_sequence
    if data_format == 'channels_last':
      conv_feats = conv_inputs  # B T F
    else:
      conv_feats = tf.transpose(conv_inputs, [0, 2, 1])  # B F T

    residual_aggregation = []

    # ----- Convolutional layers ---------------------------------------------
    convnet_layers = self.params['convnet_layers']

    for idx_convnet in range(len(convnet_layers)):
      layer_type = convnet_layers[idx_convnet]['type']
      layer_repeat = convnet_layers[idx_convnet]['repeat']
      ch_out = convnet_layers[idx_convnet]['num_channels']
      kernel_size = convnet_layers[idx_convnet]['kernel_size']
      strides = convnet_layers[idx_convnet]['stride']
      padding = convnet_layers[idx_convnet]['padding']
      dilation = convnet_layers[idx_convnet]['dilation']
      dropout_keep = convnet_layers[idx_convnet].get(
          'dropout_keep_prob', dropout_keep_prob) if training else 1.0
      residual = convnet_layers[idx_convnet].get('residual', False)
      residual_dense = convnet_layers[idx_convnet].get('residual_dense', False)

      if residual:
        layer_res = conv_feats
        if residual_dense:
          residual_aggregation.append(layer_res)
          layer_res = residual_aggregation
      for idx_layer in range(layer_repeat):
        if padding == "VALID":
          src_length = (src_length - kernel_size[0]) // strides[0] + 1
        else:
          src_length = (src_length + strides[0] - 1) // strides[0]
        if residual and idx_layer == layer_repeat - 1:
          conv_feats = conv_bn_res_bn_actv(
              layer_type=layer_type,
              name="conv{}{}".format(
                  idx_convnet + 1, idx_layer + 1),
              inputs=conv_feats,
              res_inputs=layer_res,
              filters=ch_out,
              kernel_size=kernel_size,
              activation_fn=self.params['activation_fn'],
              strides=strides,
              padding=padding,
              dilation=dilation,
              regularizer=regularizer,
              training=training,
              data_format=data_format,
              **normalization_params
          )
        else:
          conv_feats = conv_block(
              layer_type=layer_type,
              name="conv{}{}".format(
                  idx_convnet + 1, idx_layer + 1),
              inputs=conv_feats,
              filters=ch_out,
              kernel_size=kernel_size,
              activation_fn=self.params['activation_fn'],
              strides=strides,
              padding=padding,
              dilation=dilation,
              regularizer=regularizer,
              training=training,
              data_format=data_format,
              **normalization_params
          )
        conv_feats = tf.nn.dropout(x=conv_feats, keep_prob=dropout_keep)

    outputs = conv_feats

    if data_format == 'channels_first':
      outputs = tf.transpose(outputs, [0, 2, 1])

    # ----- Transformer layers -------------------------------------------

    if len(self.layers) == 0:
      for _ in range(self.params['encoder_layers']):
        # Create sublayers for each layer.
        self_attention_layer = attention_layer.SelfAttention(
          hidden_size=self.params["hidden_size"],
          num_heads=self.params["num_heads"],
          attention_dropout=self.params["attention_dropout"],
          train=training,
          regularizer=self.regularizer
        )
        feed_forward_network = ffn_layer.FeedFowardNetwork(
          hidden_size=self.params["hidden_size"],
          filter_size=self.params["filter_size"],
          relu_dropout=self.params["relu_dropout"],
          train=training,
          regularizer=self.regularizer
        )

        self.layers.append([
            PrePostProcessingWrapper(self_attention_layer, self.params,
                                     training),
            PrePostProcessingWrapper(feed_forward_network, self.params,
                                     training)
        ])

      # final normalization layer.
      print("Encoder:", self.norm_params["type"], self.mode)
      if self.norm_params["type"] =="batch_norm":
        self.output_normalization = Transformer_BatchNorm(
          training=training,
          params=self.norm_params)
      else:
        self.output_normalization = LayerNormalization(
          hidden_size=self.params["hidden_size"], params=self.norm_params)

    # actual encoder part
    with tf.name_scope("transformer_encode"):
      embedded_inputs = outputs # this is output from conv layers
      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(embedded_inputs)[1]
        pos_encoding = utils.get_position_encoding(
            length, self.params["hidden_size"],
        )
        encoder_inputs = embedded_inputs + tf.cast(x=pos_encoding,
                                                   dtype=embedded_inputs.dtype)

      if self.mode == "train":
        encoder_inputs = tf.nn.dropout(encoder_inputs,
            keep_prob = 1.0 - self.params["layer_postprocess_dropout"],
        )

      outputs = self._call(encoder_inputs)

    return {
        'outputs': outputs,
        'src_length': src_length,
    }
