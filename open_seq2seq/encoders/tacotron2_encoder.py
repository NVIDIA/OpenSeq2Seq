# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import math
import inspect

import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.framework import ops
from open_seq2seq.parts.rnns.utils import single_cell
from open_seq2seq.parts.cnns.conv_blocks import conv_bn_actv


from .encoder import Encoder

# def conv1d_bn_actv(name, inputs, filters, kernel_size, activation_fn, strides,
#                    padding, regularizer, training, use_bias, data_format, 
#                    enable_bn, bn_momentum, bn_epsilon):
#   """Helper function that applies 1-D convolution, batch norm and activation."""
#   conv = tf.layers.conv1d(
#     name="{}".format(name),
#     inputs=inputs,
#     filters=filters,
#     kernel_size=kernel_size,
#     strides=strides,
#     padding=padding,
#     kernel_regularizer=regularizer,
#     use_bias=use_bias,
#     data_format=data_format,
#   )
#   output = conv
#   if enable_bn:
#     bn = tf.layers.batch_normalization(
#       name="{}/bn".format(name),
#       inputs=conv,
#       gamma_regularizer=regularizer,
#       training=training,
#       axis=-1 if data_format == 'channels_last' else 1,
#       momentum=bn_momentum,
#       epsilon=bn_epsilon,
#     )
#     output = bn
#   if activation_fn is not None:
#     output = activation_fn(output)
#   return output

# def rnn_cell(rnn_cell_dim, layer_type, dropout_keep_prob=1.0):
#   """Helper function that creates RNN cell."""
#   if layer_type == "layernorm_lstm":
#     cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
#       num_units=rnn_cell_dim, dropout_keep_prob=dropout_keep_prob)
#   else:
#     if layer_type == "lstm":
#       cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_cell_dim)
#     elif layer_type == "gru":
#       cell = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
#     elif layer_type == "cudnn_gru":
#       cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(rnn_cell_dim)
#     elif layer_type == "cudnn_lstm":
#       cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(rnn_cell_dim)
#     else:
#       raise ValueError("Error: not supported rnn type:{}".format(layer_type))

#     cell = tf.nn.rnn_cell.DropoutWrapper(
#       cell, output_keep_prob=dropout_keep_prob)
#   return cell

class Tacotron2Encoder(Encoder):
  """Tacotron-2 like encoder. 

  Consists of an embedding layer followed by a convolutional layer followed by a recurrent layer.
  """
  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
      'dropout_keep_prob': float,
      'src_emb_size': int,
      'conv_layers': list,
      'activation_fn': None,  # any valid callable
      'enable_bn': bool,
      'num_rnn_layers': int,
      'rnn_cell_dim': int,
      'use_cudnn_rnn': bool,
      'rnn_type': None,
      'rnn_unidirectional': bool,
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
      'use_bias': bool,
      'data_format': ['channels_first', 'channels_last'],
      'bn_momentum': float,
      'bn_epsilon': float,
      'zoneout_prob': float,
    })

  def __init__(self, params, model, name="tacotron2_encoder", mode='train'):
    """Tacotron-2 like encoder constructor.

    See parent class for arguments description.

    Config parameters:

    * **dropout_keep_prop** (float) --- keep probability for dropout.
    * **src_emb_size** (int) --- dimensionality of character embedding.
    * **conv_layers** (list) --- list with the description of convolutional
      layers. For example::
        "conv_layers": [
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME"
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME"
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME"
          }
        ]
    * **activation_fn** (callable) --- activation function to use for conv layers.
    * **enable_bn** (bool) --- whether to enable batch norm after each conv layer.
    * **num_rnn_layers** --- number of RNN layers to use.
    * **rnn_cell_dim** (int) --- dimension of RNN cells.
    * **rnn_type** (callable) --- Any valid RNN Cell class. Suggested class is lstm
    * **rnn_unidirectional** (bool) --- whether to use uni-directional or
      bi-directional RNNs.
    * **use_bias** (bool) --- whether to enable a bias unit for the conv
      layers. Defaults to True
    * **data_format** (string) --- could be either "channels_first" or
      "channels_last". Defaults to "channels_last".
    * **bn_momentum** (float) --- momentum for batch norm. Defaults to 0.1.
    * **bn_epsilon** (float) --- epsilon for batch norm. Defaults to 1e-5.
    """
    super(Tacotron2Encoder, self).__init__(params, model, name, mode)

  def _encode(self, input_dict):
    """Creates TensorFlow graph for Tacotron-2 like encoder.

    Expects the following inputs::

      input_dict = {
        "src_sequence": tensor of shape [batch_size, sequence length]
        "src_length": tensor of shape [batch_size]
      }
    """

    source_sequence, src_length = input_dict['source_tensors']

    training = (self._mode == "train")
    dropout_keep_prob = self.params['dropout_keep_prob'] if training else 1.0
    regularizer = self.params.get('regularizer', None)
    # use_bias = self.params.get('use_bias', True)
    data_format = self.params.get('data_format', 'channels_last')
    bn_momentum = self.params.get('bn_momentum', 0.1)
    bn_epsilon = self.params.get('bn_epsilon', 1e-5)
    src_vocab_size = self._model.get_data_layer().params['src_vocab_size']
    zoneout_prob = self.params.get('zoneout_prob', 0.)

    # enable_bn = self.params.get('enable_bn', True)
    # src_emb_size = self.params.get('src_emb_size', 512)

    # ----- Embedding layer -----------------------------------------------
    enc_emb_w = tf.get_variable(
      name="EncoderEmbeddingMatrix",
      shape=[src_vocab_size, self.params['src_emb_size']],
      dtype=self.params['dtype'],
      # initializer=tf.random_normal_initializer()
    )

    embedded_inputs = tf.cast(tf.nn.embedding_lookup(
      enc_emb_w,
      source_sequence,
    ), self.params['dtype'])

    # ----- Convolutional layers -----------------------------------------------
    input_layer = embedded_inputs

    batch_size = input_layer.get_shape().as_list()[0]

    if data_format == 'channels_last':
      top_layer = input_layer
    else:
      top_layer = tf.transpose(input_layer, [0, 2, 1])

    conv_layers = self.params['conv_layers']

    for idx_conv in range(len(conv_layers)):
      ch_out = conv_layers[idx_conv]['num_channels']
      kernel_size = conv_layers[idx_conv]['kernel_size']  # [time, freq]
      strides = conv_layers[idx_conv]['stride']
      padding = conv_layers[idx_conv]['padding']

      if padding == "VALID":
        src_length = (src_length - kernel_size[0] + strides[0]) // strides[0]
      else:
        src_length = (src_length + strides[0] - 1) // strides[0]

      top_layer = conv_bn_actv(
        type="conv1d",
        name="conv{}".format(idx_conv + 1),
        inputs=top_layer,
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
      top_layer = tf.layers.dropout(top_layer, rate=1.-dropout_keep_prob, training=training)

    if data_format == 'channels_first':
      top_layer = tf.transpose(top_layer, [0, 2, 1])

    # ----- RNN ---------------------------------------------------------------
    num_rnn_layers = self.params['num_rnn_layers']
    # Disable dropout for rnn layers, need to switch to zoneout
    dropout_keep_prob = 1.
    if num_rnn_layers > 0:
      cell_params = {}
      cell_params["num_units"] = self.params['rnn_cell_dim']
      rnn_type = self.params['rnn_type']
      rnn_input = top_layer
      rnn_vars = []

      if self.params["use_cudnn_rnn"]:
        all_cudnn_classes = [i[1] for i in inspect.getmembers(tf.contrib.cudnn_rnn, inspect.isclass)]
        if not rnn_type in all_cudnn_classes:
          raise TypeError("rnn_type must be a Cudnn RNN class")
        if zoneout_prob != 0.:
          raise ValueError("Zoneout is currently not supported for cudnn rnn classes")

        rnn_input = tf.transpose(top_layer, [1, 0, 2])
        if self.params['rnn_unidirectional']:
          direction = cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION
        else:
          direction = cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION

        rnn_block = rnn_type(
          num_layers = num_rnn_layers,
          num_units = cell_params["num_units"],
          direction = direction,
          dtype=rnn_input.dtype,
          name="cudnn_rnn"
          )
        top_layer, state = rnn_block(rnn_input)
        top_layer = tf.transpose(top_layer, [1, 0, 2])
        rnn_vars += rnn_block.trainable_variables

      else:
        multirnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
          [single_cell(cell_class=rnn_type,
                       cell_params=cell_params,
                       # dp_input_keep_prob=dropout_keep_prob,
                       # dp_output_keep_prob=dropout_keep_prob,
                       zoneout_prob=zoneout_prob,
                       training=training,
                       residual_connections=False)
           for _ in range(num_rnn_layers)]
        )
        rnn_vars += multirnn_cell_fw.trainable_variables
        if self.params['rnn_unidirectional']:
          top_layer, state = tf.nn.dynamic_rnn(
            cell=multirnn_cell_fw,
            inputs=rnn_input,
            sequence_length=src_length,
            dtype=rnn_input.dtype,
            time_major=False,
          )
        else:
          multirnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
            [single_cell(cell_class=rnn_type,
                         cell_params=cell_params,
                         # dp_input_keep_prob=dropout_keep_prob,
                         # dp_output_keep_prob=dropout_keep_prob,
                         zoneout_prob=zoneout_prob,
                         training=training,
                         residual_connections=False)
             for _ in range(num_rnn_layers)]
          )
          top_layer, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=multirnn_cell_fw, cell_bw=multirnn_cell_bw,
            inputs=rnn_input,
            sequence_length=src_length,
            dtype=rnn_input.dtype,
            time_major=False
          )
          # concat 2 tensors [B, T, n_cell_dim] --> [B, T, 2*n_cell_dim]
          top_layer = tf.concat(top_layer, 2)
          rnn_vars += multirnn_cell_bw.trainable_variables

      if regularizer and training:
        cell_weights = []
        cell_weights += rnn_vars
        cell_weights += [enc_emb_w]
        for weights in cell_weights:
          if "bias" not in weights.name:
            print("Added regularizer to {}".format(weights.name))
            if weights.dtype.base_dtype == tf.float16:
              tf.add_to_collection('REGULARIZATION_FUNCTIONS', (weights, regularizer))
            else:
              tf.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, regularizer(weights))
          # Want to change init for bias
          # else:
          #   std = 1.0 / math.sqrt(self.params['rnn_cell_dim'])
          #   weights.initializer = tf.random_uniform(weights.get_shape(), minval=-std, maxval=std)

    # -- end of rnn------------------------------------------------------------

    outputs = top_layer
    
    return {
      'outputs': outputs,
      'src_length': src_length,
      # 'state': state
    }
