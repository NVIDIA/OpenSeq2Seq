# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf

from .encoder import Encoder
from open_seq2seq.parts.cnns.conv_blocks import conv_bn_actv 

class Wave2LetterEncoder(Encoder):
  """Wave2Letter like encoder."""
  """Fully convolutional model"""

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
      'bn_momentum': float,
      'bn_epsilon': float,
    })

  def __init__(self, params, model, name="w2l_encoder", mode='train'):
    """Wave2Letter like encoder constructor.

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
    * **bn_momentum** (float) --- momentum for batch norm. Defaults to 0.99.
    * **bn_epsilon** (float) --- epsilon for batch norm. Defaults to 1e-3.
    """
    super(Wave2LetterEncoder, self).__init__(params, model, name, mode)

  def _get_layer(self, layer_type):
    if layer_type == "conv1d":
      return "conv", tf.layers.conv1d
    elif layer_type == "conv2d":
      return "conv", tf.layers.conv2d

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
    bn_momentum = self.params.get('bn_momentum', 0.99)
    bn_epsilon = self.params.get('bn_epsilon', 1e-3)

    conv_inputs = source_sequence
    batch_size = conv_inputs.get_shape().as_list()[0]
    if data_format == 'channels_last':
      conv_feats = conv_inputs #B T F
    else:
      conv_feats = tf.transpose(conv_inputs, [0, 2, 1]) #B F T

    # ----- Convolutional layers -----------------------------------------------
    convnet_layers = self.params['convnet_layers']

    for idx_convnet in range(len(convnet_layers)):
      layer_type = convnet_layers[idx_convnet]['type']
      layer_repeat_fixed = convnet_layers[idx_convnet]['repeat']
      layer_repeat_moving = layer_repeat_fixed

      while(layer_repeat_moving != 0):
        layer_repeat_moving = layer_repeat_moving -1
        layer_name, layer = self._get_layer(layer_type)
        if layer_name == "conv":
          ch_out = convnet_layers[idx_convnet]['num_channels']
          conv_block = conv_bn_actv #can add other type of convolutional blocks in future
          kernel_size = convnet_layers[idx_convnet]['kernel_size']
          strides = convnet_layers[idx_convnet]['stride']
          padding = convnet_layers[idx_convnet]['padding']

          conv_feats = conv_block(
            layer = layer,
            name="conv{}{}".format(idx_convnet + 1, layer_repeat_fixed + 1 - layer_repeat_moving),
            inputs=conv_feats,
            filters=ch_out,
            kernel_size=kernel_size,
            activation_fn=self.params['activation_fn'],
            strides=strides,
            padding=padding,
            regularizer=regularizer,
            training=training,
            data_format=data_format,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
          )

          outputs = tf.nn.dropout(x=conv_feats, keep_prob=dropout_keep_prob)

    if data_format == 'channels_first':
      outputs = tf.transpose(outputs, [0, 2, 1])

    return {
      'outputs': outputs,
      'src_length': src_length,
    }
