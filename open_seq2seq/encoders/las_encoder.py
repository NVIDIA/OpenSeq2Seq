# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from .encoder import Encoder

cells_dict = {
    "lstm": tf.nn.rnn_cell.BasicLSTMCell,
    "gru": tf.nn.rnn_cell.GRUCell,
}


def rnn_layer(layer_type, num_layers, name, inputs, src_length, hidden_dim,
              regularizer, training, dropout_keep_prob=1.0):
  """Helper function that applies convolution and activation.
    Args:
      layer_type: the following types are supported
        'lstm', 'gru'
  """
  rnn_cell = cells_dict[layer_type]
  dropout = tf.nn.rnn_cell.DropoutWrapper

  multirnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
      [dropout(rnn_cell(hidden_dim),
               output_keep_prob=dropout_keep_prob)
       for _ in range(num_layers)]
  )

  multirnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
      [dropout(rnn_cell(hidden_dim),
               output_keep_prob=dropout_keep_prob)
       for _ in range(num_layers)]
  )

  output, state = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=multirnn_cell_fw, cell_bw=multirnn_cell_bw,
      inputs=inputs,
      sequence_length=src_length,
      dtype=inputs.dtype,
      scope=name,
  )
  output = tf.concat(output, 2)

  return output


class ListenAttendSpellEncoder(Encoder):
  """Listen Attend Spell like encoder"""

  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
        'dropout_keep_prob': float,
        'recurrent_layers': list,
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
        'residual_connections': bool,
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
    * **normalization** --- normalization to use. Accepts [None, 'batch_norm'].
      Use None if you don't want to use normalization. Defaults to 'batch_norm'.     
    * **bn_momentum** (float) --- momentum for batch norm. Defaults to 0.90.
    * **bn_epsilon** (float) --- epsilon for batch norm. Defaults to 1e-3.
    """
    super(ListenAttendSpellEncoder, self).__init__(params, model, name, mode)

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

    rnn_feats = source_sequence
    rnn_block = rnn_layer

    # ----- Recurrent layers ---------------------------------------------
    recurrent_layers = self.params['recurrent_layers']

    for idx_rnn in range(len(recurrent_layers)):
      layer_type = recurrent_layers[idx_rnn]['type']
      num_layers = recurrent_layers[idx_rnn]['num_layers']
      hidden_dim = recurrent_layers[idx_rnn]['hidden_dim']
      dropout_keep = recurrent_layers[idx_rnn].get(
          'dropout_keep_prob', dropout_keep_prob) if training else 1.0

      rnn_feats = rnn_block(
          layer_type=layer_type,
          num_layers=num_layers,
          name="rnn{}".format(
              idx_rnn + 1),
          inputs=rnn_feats,
          src_length=src_length,
          hidden_dim=hidden_dim,
          regularizer=regularizer,
          training=training,
          dropout_keep_prob=dropout_keep,
      )

    outputs = rnn_feats

    return {
        'outputs': outputs,
        'src_length': src_length,
    }
