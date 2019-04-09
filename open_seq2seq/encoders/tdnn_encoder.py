# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from .encoder import Encoder
from open_seq2seq.data.speech2text.speech2text import Speech2TextDataLayer
from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv,\
                                                conv_ln_actv, conv_in_actv,\
                                                conv_bn_res_bn_actv


class TDNNEncoder(Encoder):
  """General time delay neural network (TDNN) encoder. Fully convolutional model
  """

  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
        'dropout_keep_prob': float,
        'convnet_layers': list,
        'activation_fn': None,  # any valid callable
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
        'data_format': ['channels_first', 'channels_last'],
        'normalization': [None, 'batch_norm', 'layer_norm', 'instance_norm'],
        'bn_momentum': float,
        'bn_epsilon': float,
        'use_conv_mask': bool,
        'drop_block_prob': float,
        'drop_block_index': int,
    })

  def __init__(self, params, model, name="w2l_encoder", mode='train'):
    """TDNN encoder constructor.

    See parent class for arguments description.

    Config parameters:

    * **dropout_keep_prob** (float) --- keep probability for dropout.
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
    * **drop_block_prob** (float) --- probability of dropping encoder blocks.
      Defaults to 0.0 which corresponds to training without dropping blocks.
    * **drop_block_index** (int) -- index of the block to drop on inference.
      Defaults to -1 which corresponds to keeping all blocks.
    * **use_conv_mask** (bool) --- whether to apply a sequence mask prior to
      convolution operations. Defaults to False for backwards compatibility.
      Recommended to set as True
    """
    super(TDNNEncoder, self).__init__(params, model, name, mode)

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
    
    num_pad = tf.constant(0)

    if isinstance(self._model.get_data_layer(), Speech2TextDataLayer):
      pad_to = 0
      if self._model.get_data_layer().params.get('backend', 'psf') == 'librosa':
        pad_to = self._model.get_data_layer().params.get("pad_to", 8)
        
      if pad_to > 0:
        num_pad = tf.mod(pad_to - tf.mod(tf.reduce_max(src_length), pad_to), pad_to)
    else:
      print("WARNING: TDNNEncoder is currently meant to be used with the",
            "Speech2Text data layer. Assuming that this data layer does not",
            "do additional padding past padded_batch.")

    max_len = tf.reduce_max(src_length) + num_pad

    training = (self._mode == "train")
    dropout_keep_prob = self.params['dropout_keep_prob'] if training else 1.0
    regularizer = self.params.get('regularizer', None)
    data_format = self.params.get('data_format', 'channels_last')
    normalization = self.params.get('normalization', 'batch_norm')

    drop_block_prob = self.params.get('drop_block_prob', 0.0)
    drop_block_index = self.params.get('drop_block_index', -1)

    normalization_params = {}

    if self.params.get("use_conv_mask", False):
      mask = tf.sequence_mask(
          lengths=src_length, maxlen=max_len,
          dtype=source_sequence.dtype
      )
      mask = tf.expand_dims(mask, 2)

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


      # For the first layer in the block, apply a mask
      if self.params.get("use_conv_mask", False):
        conv_feats = conv_feats * mask

      if residual:
        layer_res = conv_feats
        if residual_dense:
          residual_aggregation.append(layer_res)
          layer_res = residual_aggregation

      for idx_layer in range(layer_repeat):

        if padding == "VALID":
          src_length = (src_length - kernel_size[0]) // strides[0] + 1
          max_len = (max_len - kernel_size[0]) // strides[0] + 1
        else:
          src_length = (src_length + strides[0] - 1) // strides[0]
          max_len = (max_len + strides[0] - 1) // strides[0]

        # For all layers other than first layer, apply mask
        if idx_layer > 0 and self.params.get("use_conv_mask", False):
          conv_feats = conv_feats * mask

        # Since we have a stride 2 layer, we need to update mask for future operations
        if (self.params.get("use_conv_mask", False) and
            (padding == "VALID" or strides[0] > 1)):
          mask = tf.sequence_mask(
              lengths=src_length,
              maxlen=max_len,
              dtype=conv_feats.dtype
          )
          mask = tf.expand_dims(mask, 2)

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
              drop_block_prob=drop_block_prob,
              drop_block=(drop_block_index == idx_convnet),
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

    return {
        'outputs': outputs,
        'src_length': src_length,
    }
