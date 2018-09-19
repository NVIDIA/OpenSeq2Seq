# Copyright (c) 2018 NVIDIA Corporation
"""
Tacotron2 decoder
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.python.framework import ops

from open_seq2seq.parts.rnns.utils import single_cell
from open_seq2seq.parts.rnns.attention_wrapper import BahdanauAttention, \
                                                 LocationSensitiveAttention, \
                                                 AttentionWrapper
from open_seq2seq.parts.tacotron.tacotron_helper import TacotronHelper, \
                                                        TacotronTrainingHelper
from open_seq2seq.parts.tacotron.tacotron_decoder import TacotronDecoder
from open_seq2seq.parts.cnns.conv_blocks import conv_bn_actv
from .decoder import Decoder


class Prenet():
  """
  Fully connected prenet used in the decoder
  """
  def __init__(
      self,
      num_units,
      num_layers,
      activation_fn=None,
      dtype=None
  ):
    """Prenet initializer

    Args:
      num_units (int): number of units in the fully connected layer
      num_layers (int): number of fully connected layers
      activation_fn (callable): any valid activation function
      dtype (dtype): the data format for this layer
    """
    assert (
        num_layers > 0
    ), "If the prenet is enabled, there must be at least 1 layer"
    self.prenet_layers = []
    self._output_size = num_units

    for idx in range(num_layers):
      self.prenet_layers.append(
          tf.layers.Dense(
              name="prenet_{}".format(idx + 1),
              units=num_units,
              activation=activation_fn,
              use_bias=True,
              dtype=dtype
          )
      )

  def __call__(self, inputs):
    """
    Applies the prenet to the inputs
    """
    for layer in self.prenet_layers:
      inputs = tf.layers.dropout(layer(inputs), rate=0.5, training=True)
    return inputs

  @property
  def output_size(self):
    return self._output_size

  def add_regularization(self, regularizer):
    """
    Adds regularization to all prenet kernels
    """
    for layer in self.prenet_layers:
      for weights in layer.trainable_variables:
        if "bias" not in weights.name:
          # print("Added regularizer to {}".format(weights.name))
          if weights.dtype.base_dtype == tf.float16:
            tf.add_to_collection(
                'REGULARIZATION_FUNCTIONS', (weights, regularizer)
            )
          else:
            tf.add_to_collection(
                ops.GraphKeys.REGULARIZATION_LOSSES, regularizer(weights)
            )


class Tacotron2Decoder(Decoder):
  """
  Tacotron 2 Decoder
  """

  @staticmethod
  def get_required_params():
    return dict(
        Decoder.get_required_params(), **{
            'attention_layer_size': int,
            'attention_type': ['bahdanau', 'location', None],
            'decoder_cell_units': int,
            'decoder_cell_type': None,
            'decoder_layers': int,
        }
    )

  @staticmethod
  def get_optional_params():
    return dict(
        Decoder.get_optional_params(), **{
            'bahdanau_normalize': bool,
            'time_major': bool,
            'use_swap_memory': bool,
            'enable_prenet': bool,
            'prenet_layers': int,
            'prenet_units': int,
            'prenet_activation': None,
            'enable_postnet': bool,
            'postnet_conv_layers': list,
            'postnet_bn_momentum': float,
            'postnet_bn_epsilon': float,
            'postnet_data_format': ['channels_first', 'channels_last'],
            'postnet_keep_dropout_prob': float,
            'mask_decoder_sequence': bool,
            'attention_bias': bool,
            'zoneout_prob': float,
            'dropout_prob': float,
            'parallel_iterations': int,
        }
    )

  def __init__(self, params, model, name='tacotron_2_decoder', mode='train'):
    """Tacotron-2 like decoder constructor. A lot of optional configurations are
    currently for testing. Not all configurations are supported. Use of thed
    efault config is reccommended.

    See parent class for arguments description.

    Config parameters:

    * **attention_layer_size** (int) --- size of attention layer.
    * **attention_type** (string) --- Determines whether attention mechanism to
      use, should be one of 'bahdanau', 'location', or None.
      Use of 'location'-sensitive attention is strongly recommended.
    * **bahdanau_normalize** (bool) --- Whether to enable weight norm on the
      attention parameters. Defaults to False.
    * **decoder_cell_units** (int) --- dimension of decoder RNN cells.
    * **decoder_layers** (int) --- number of decoder RNN layers to use.
    * **decoder_cell_type** (callable) --- could be "lstm", "gru", "glstm", or
      "slstm". Currently, only 'lstm' has been tested. Defaults to 'lstm'.
    * **time_major** (bool) --- whether to output as time major or batch major.
      Default is False for batch major.
    * **use_swap_memory** (bool) --- default is False.
    * **enable_prenet** (bool) --- whether to use the fully-connected prenet in
      the decoder. Defaults to True
    * **prenet_layers** (int) --- number of fully-connected layers to use.
      Defaults to 2.
    * **prenet_units** (int) --- number of units in each layer. Defaults to 256.
    * **prenet_activation** (callable) --- activation function to use for the
      prenet lyaers. Defaults to relu
    * **enable_postnet** (bool) --- whether to use the convolutional postnet in
      the decoder. Defaults to True
    * **postnet_conv_layers** (bool) --- list with the description of
      convolutional layers. Must be passed if postnet is enabled
      For example::
        "postnet_conv_layers": [
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME",
            "activation_fn": tf.nn.tanh
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME",
            "activation_fn": tf.nn.tanh
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME",
            "activation_fn": tf.nn.tanh
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME",
            "activation_fn": tf.nn.tanh
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 80, "padding": "SAME",
            "activation_fn": None
          }
        ]
    * **postnet_bn_momentum** (float) --- momentum for batch norm.
      Defaults to 0.1.
    * **postnet_bn_epsilon** (float) --- epsilon for batch norm.
      Defaults to 1e-5.
    * **postnet_data_format** (string) --- could be either "channels_first" or
      "channels_last". Defaults to "channels_last".
    * **postnet_keep_dropout_prob** (float) --- keep probability for dropout in
      the postnet conv layers. Default to 0.5.
    * **mask_decoder_sequence** (bool) --- Defaults to True.
    * **attention_bias** (bool) --- Wether to use a bias term when calculating
      the attention. Only works for "location" attention. Defaults to False.
    * **zoneout_prob** (float) --- zoneout probability for rnn layers.
      Defaults to 0.
    * **dropout_prob** (float) --- dropout probability for rnn layers.
      Defaults to 0.1
    * **parallel_iterations** (int) --- Number of parallel_iterations for
      tf.while loop inside dynamic_decode. Defaults to 32.
    """

    super(Tacotron2Decoder, self).__init__(params, model, name, mode)
    self._model = model
    self._n_feats = self._model.get_data_layer().params['num_audio_features']
    if "both" in self._model.get_data_layer().params['output_type']:
      self._both = True
      if not self.params.get('enable_postnet', True):
        raise ValueError(
            "postnet must be enabled for both mode"
        )
    else:
      self._both = False

  def _build_attention(
      self,
      encoder_outputs,
      encoder_sequence_length,
      attention_bias,
  ):
    """
    Builds Attention part of the graph.
    Currently supports "bahdanau", and "location"
    """
    with tf.variable_scope("AttentionMechanism"):
      attention_depth = self.params['attention_layer_size']
      if self.params['attention_type'] == 'location':
        attention_mechanism = LocationSensitiveAttention(
            num_units=attention_depth,
            memory=encoder_outputs,
            memory_sequence_length=encoder_sequence_length,
            probability_fn=tf.nn.softmax,
            dtype=tf.get_variable_scope().dtype,
            use_bias=attention_bias,
        )
      elif self.params['attention_type'] == 'bahdanau':
        bah_normalize = self.params.get('bahdanau_normalize', False)
        attention_mechanism = BahdanauAttention(
            num_units=attention_depth,
            memory=encoder_outputs,
            normalize=bah_normalize,
            memory_sequence_length=encoder_sequence_length,
            probability_fn=tf.nn.softmax,
            dtype=tf.get_variable_scope().dtype
        )
      else:
        raise ValueError('Unknown Attention Type')
      return attention_mechanism

  def _decode(self, input_dict):
    """
    Decodes representation into data

    Args:
      input_dict (dict): Python dictionary with inputs to decoder. Must define:
          * src_inputs - decoder input Tensor of shape [batch_size, time, dim]
            or [time, batch_size, dim]
          * src_lengths - decoder input lengths Tensor of shape [batch_size]
          * tgt_inputs - Only during training. labels Tensor of the
            shape [batch_size, time, num_features] or
            [time, batch_size, num_features]
          * stop_token_inputs - Only during training. labels Tensor of the
            shape [batch_size, time, 1] or [time, batch_size, 1]
          * tgt_lengths - Only during training. labels lengths
            Tensor of the shape [batch_size]

    Returns:
      dict:
        A python dictionary containing:

          * outputs - array containing:

              * decoder_output - tensor of shape [batch_size, time,
                num_features] or [time, batch_size, num_features]. Spectrogram
                representation learned by the decoder rnn
              * spectrogram_prediction - tensor of shape [batch_size, time,
                num_features] or [time, batch_size, num_features]. Spectrogram
                containing the residual corrections from the postnet if enabled
              * alignments - tensor of shape [batch_size, time, memory_size]
                or [time, batch_size, memory_size]. The alignments learned by
                the attention layer
              * stop_token_prediction - tensor of shape [batch_size, time, 1]
                or [time, batch_size, 1]. The stop token predictions
              * final_sequence_lengths - tensor of shape [batch_size]
          * stop_token_predictions - tensor of shape [batch_size, time, 1]
            or [time, batch_size, 1]. The stop token predictions for use inside
            the loss function.
    """
    encoder_outputs = input_dict['encoder_output']['outputs']
    enc_src_lengths = input_dict['encoder_output']['src_length']
    if self._mode == "train":
      spec = input_dict['target_tensors'][0] if 'target_tensors' in \
                                                    input_dict else None
      spec_length = input_dict['target_tensors'][2] if 'target_tensors' in \
                                                    input_dict else None
    _batch_size = encoder_outputs.get_shape().as_list()[0]

    training = (self._mode == "train")
    regularizer = self.params.get('regularizer', None)

    if self.params.get('enable_postnet', True):
      if "postnet_conv_layers" not in self.params:
        raise ValueError(
            "postnet_conv_layers must be passed from config file if postnet is"
            "enabled"
        )

    if self._both:
      num_audio_features = self._n_feats["mel"]
      if self._mode == "train":
        spec, _ = tf.split(
            spec,
            [self._n_feats['mel'], self._n_feats['magnitude']],
            axis=2
        )
    else:
      num_audio_features = self._n_feats

    output_projection_layer = tf.layers.Dense(
        name="output_proj",
        units=num_audio_features,
        use_bias=True,
    )
    stop_token_projection_layer = tf.layers.Dense(
        name="stop_token_proj",
        units=1,
        use_bias=True,
    )

    prenet = None
    if self.params.get('enable_prenet', True):
      prenet = Prenet(
          self.params.get('prenet_units', 256),
          self.params.get('prenet_layers', 2),
          self.params.get("prenet_activation", tf.nn.relu),
          self.params["dtype"]
      )

    cell_params = {}
    cell_params["num_units"] = self.params['decoder_cell_units']
    decoder_cells = [
        single_cell(
            cell_class=self.params['decoder_cell_type'],
            cell_params=cell_params,
            zoneout_prob=self.params.get("zoneout_prob", 0.),
            dp_output_keep_prob=1.-self.params.get("dropout_prob", 0.1),
            training=training,
        ) for _ in range(self.params['decoder_layers'])
    ]

    if self.params['attention_type'] is not None:
      attention_mechanism = self._build_attention(
          encoder_outputs, enc_src_lengths,
          self.params.get("attention_bias", False)
      )

      attention_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)

      attentive_cell = AttentionWrapper(
          cell=attention_cell,
          attention_mechanism=attention_mechanism,
          alignment_history=True,
          output_attention="both",
      )

      decoder_cell = attentive_cell

    if self.params['attention_type'] is None:
      decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)

    if self._mode == "train":
      train_and_not_sampling = True
      helper = TacotronTrainingHelper(
          inputs=spec,
          sequence_length=spec_length,
          prenet=None,
          model_dtype=self.params["dtype"],
          mask_decoder_sequence=self.params.get("mask_decoder_sequence", True)
      )
    elif self._mode == "eval" or self._mode == "infer":
      train_and_not_sampling = False
      inputs = tf.zeros(
          (_batch_size, 1, num_audio_features), dtype=self.params["dtype"]
      )
      helper = TacotronHelper(
          inputs=inputs,
          prenet=None,
          mask_decoder_sequence=self.params.get("mask_decoder_sequence", True)
      )
    else:
      raise ValueError("Unknown mode for decoder: {}".format(self._mode))
    decoder = TacotronDecoder(
        decoder_cell=decoder_cell,
        helper=helper,
        initial_decoder_state=decoder_cell.zero_state(
            _batch_size, self.params["dtype"]
        ),
        attention_type=self.params["attention_type"],
        spec_layer=output_projection_layer,
        stop_token_layer=stop_token_projection_layer,
        prenet=prenet,
        dtype=self.params["dtype"],
        train=train_and_not_sampling
    )

    if self._mode == 'train':
      maximum_iterations = tf.reduce_max(spec_length)
    else:
      maximum_iterations = tf.reduce_max(enc_src_lengths) * 10

    outputs, final_state, sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
        # outputs, final_state, sequence_lengths, final_inputs = dynamic_decode(
        decoder=decoder,
        impute_finished=False,
        maximum_iterations=maximum_iterations,
        swap_memory=self.params.get("use_swap_memory", False),
        output_time_major=self.params.get("time_major", False),
        parallel_iterations=self.params.get("parallel_iterations", 32)
    )

    decoder_output = outputs.rnn_output
    stop_token_logits = outputs.stop_token_output

    with tf.variable_scope("decoder"):
      # If we are in train and doing sampling, we need to do the projections
      if train_and_not_sampling:
        decoder_spec_output = output_projection_layer(decoder_output)
        stop_token_logits = stop_token_projection_layer(decoder_spec_output)
        decoder_output = decoder_spec_output

    ## Add the post net ##
    if self.params.get('enable_postnet', True):
      dropout_keep_prob = self.params.get('postnet_keep_dropout_prob', 0.5)

      top_layer = decoder_output
      for i, conv_params in enumerate(self.params['postnet_conv_layers']):
        ch_out = conv_params['num_channels']
        kernel_size = conv_params['kernel_size']  # [time, freq]
        strides = conv_params['stride']
        padding = conv_params['padding']
        activation_fn = conv_params['activation_fn']

        if ch_out == -1:
          if self._both:
            ch_out = self._n_feats["mel"]
          else:
            ch_out = self._n_feats

        top_layer = conv_bn_actv(
            layer_type="conv1d",
            name="conv{}".format(i + 1),
            inputs=top_layer,
            filters=ch_out,
            kernel_size=kernel_size,
            activation_fn=activation_fn,
            strides=strides,
            padding=padding,
            regularizer=regularizer,
            training=training,
            data_format=self.params.get('postnet_data_format', 'channels_last'),
            bn_momentum=self.params.get('postnet_bn_momentum', 0.1),
            bn_epsilon=self.params.get('postnet_bn_epsilon', 1e-5),
        )
        top_layer = tf.layers.dropout(
            top_layer, rate=1. - dropout_keep_prob, training=training
        )

    else:
      top_layer = tf.zeros(
          [
              _batch_size, maximum_iterations,
              outputs.rnn_output.get_shape()[-1]
          ],
          dtype=self.params["dtype"]
      )

    if regularizer and training:
      vars_to_regularize = []
      vars_to_regularize += attentive_cell.trainable_variables
      vars_to_regularize += attention_mechanism.memory_layer.trainable_variables
      vars_to_regularize += output_projection_layer.trainable_variables
      vars_to_regularize += stop_token_projection_layer.trainable_variables

      for weights in vars_to_regularize:
        if "bias" not in weights.name:
          # print("Added regularizer to {}".format(weights.name))
          if weights.dtype.base_dtype == tf.float16:
            tf.add_to_collection(
                'REGULARIZATION_FUNCTIONS', (weights, regularizer)
            )
          else:
            tf.add_to_collection(
                ops.GraphKeys.REGULARIZATION_LOSSES, regularizer(weights)
            )

      if self.params.get('enable_prenet', True):
        prenet.add_regularization(regularizer)

    if self.params['attention_type'] is not None:
      alignments = tf.transpose(
          final_state.alignment_history.stack(), [1, 0, 2]
      )
    else:
      alignments = tf.zeros([_batch_size, _batch_size, _batch_size])

    spectrogram_prediction = decoder_output + top_layer
    if self._both:
      mag_spec_prediction = spectrogram_prediction
      mag_spec_prediction = conv_bn_actv(
          layer_type="conv1d",
          name="conv_0",
          inputs=mag_spec_prediction,
          filters=256,
          kernel_size=4,
          activation_fn=tf.nn.relu,
          strides=1,
          padding="SAME",
          regularizer=regularizer,
          training=training,
          data_format=self.params.get('postnet_data_format', 'channels_last'),
          bn_momentum=self.params.get('postnet_bn_momentum', 0.1),
          bn_epsilon=self.params.get('postnet_bn_epsilon', 1e-5),
      )
      mag_spec_prediction = conv_bn_actv(
          layer_type="conv1d",
          name="conv_1",
          inputs=mag_spec_prediction,
          filters=512,
          kernel_size=4,
          activation_fn=tf.nn.relu,
          strides=1,
          padding="SAME",
          regularizer=regularizer,
          training=training,
          data_format=self.params.get('postnet_data_format', 'channels_last'),
          bn_momentum=self.params.get('postnet_bn_momentum', 0.1),
          bn_epsilon=self.params.get('postnet_bn_epsilon', 1e-5),
      )
      if self._model.get_data_layer()._exp_mag:
        mag_spec_prediction = tf.exp(mag_spec_prediction)
      mag_spec_prediction = tf.layers.conv1d(
          mag_spec_prediction,
          self._n_feats["magnitude"],
          1,
          name="post_net_proj",
          use_bias=False,
      )
    else:
      mag_spec_prediction = tf.zeros([_batch_size, _batch_size, _batch_size])

    stop_token_prediction = tf.sigmoid(stop_token_logits)
    outputs = [
        decoder_output, spectrogram_prediction, alignments,
        stop_token_prediction, sequence_lengths, mag_spec_prediction
    ]

    return {
        'outputs': outputs,
        'stop_token_prediction': stop_token_logits,
    }
