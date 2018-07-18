# Copyright (c) 2018 NVIDIA Corporation
"""
This module contains classes and functions to build "general" convolutional
neural networks from the description of arbitrary "layers".
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import copy

import tensorflow as tf

try:
  from inspect import signature
except ImportError:
  from funcsigs import signature

from open_seq2seq.utils.utils import deco_print
from .encoder import Encoder


def build_layer(inputs, layer, layer_params, data_format,
                regularizer, training, verbose=True):
  """This function builds a layer from the layer function and it's parameters.

  It will automatically add regularizer parameter to the layer_params if the
  layer supports regularization. To check this, it will look for the
  "regularizer", "kernel_regularizer" and "gamma_regularizer" names in this
  order in the ``layer`` call signature. If one of this parameters is supported
  it will pass regularizer object as a value for that parameter. Based on the
  same "checking signature" technique "data_format" and "training" parameters
  will try to be added. Finally, "axis" parameter will try to be specified with
  axis = ``1 if data_format == 'channels_first' else 3``. This is required for
  automatic building batch normalization layer.

  Args:
    inputs: input Tensor that will be passed to the layer. Note that layer has
        to accept input as the first parameter.
    layer: layer function or class with ``__call__`` method defined.
    layer_params (dict): parameters passed to the ``layer``.
    data_format (string): data format ("channels_first" or "channels_last")
        that will be tried to be passed as an additional argument.
    regularizer: regularizer instance that will be tried to be passed as an
        additional argument.
    training (bool): whether layer is built in training mode. Will be tried to
        be passed as an additional argument.
    verbose (bool): whether to print information about built layers.

  Returns:
    Tensor with layer output.
  """
  layer_params_cp = copy.deepcopy(layer_params)
  for reg_name in ['regularizer', 'kernel_regularizer', 'gamma_regularizer']:
    if reg_name not in layer_params_cp and \
       reg_name in signature(layer).parameters:
      layer_params_cp.update({reg_name: regularizer})

  if 'data_format' not in layer_params_cp and \
     'data_format' in signature(layer).parameters:
    layer_params_cp.update({'data_format': data_format})

  # necessary to check axis for correct batch normalization processing
  if 'axis' not in layer_params_cp and \
     'axis' in signature(layer).parameters:
    layer_params_cp.update({'axis': 1 if data_format == 'channels_first' else 3})

  if 'training' not in layer_params_cp and \
     'training' in signature(layer).parameters:
    layer_params_cp.update({'training': training})

  outputs = layer(inputs, **layer_params_cp)

  if verbose:
    if hasattr(layer, '_tf_api_names'):
      layer_name = layer._tf_api_names[0]
    else:
      layer_name = layer
    deco_print("Building layer: {}(inputs, {})".format(
        layer_name,
        ", ".join("{}={}".format(key, value)
                  for key, value in layer_params_cp.items())
    ))
  return outputs


class CNNEncoder(Encoder):
  """General CNN encoder that can be used to construct various different models.
  """
  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
        'cnn_layers': list,
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
        'data_format': ['channels_first', 'channels_last'],
        'fc_layers': list,
    })

  def __init__(self, params, model, name="cnn_encoder", mode='train'):
    """CNN Encoder constructor.

    See parent class for arguments description.

    Config parameters:

    * **cnn_layers** (list) --- list with the description of "convolutional"
      layers. For example::
        "conv_layers": [
            (tf.layers.conv2d, {
                'filters': 64, 'kernel_size': (11, 11),
                'strides': (4, 4), 'padding': 'VALID',
                'activation': tf.nn.relu,
            }),
            (tf.layers.max_pooling2d, {
                'pool_size': (3, 3), 'strides': (2, 2),
            }),
            (tf.layers.conv2d, {
                'filters': 192, 'kernel_size': (5, 5),
                'strides': (1, 1), 'padding': 'SAME',
            }),
            (tf.layers.batch_normalization, {'momentum': 0.9, 'epsilon': 0.0001}),
            (tf.nn.relu, {}),
        ]
      Note that you don't need to provide "regularizer", "training",
      "data_format" and "axis" parameters since they will be
      automatically added. "axis" will be derived from "data_format" and will
      be ``1 if data_format == "channels_first" else 3``.

    * **fc_layers** (list) --- list with the description of "fully-connected"
      layers. The only different from convolutional layers is that the input
      will be automatically reshaped to 2D (batch size x num features).
      For example::
        'fc_layers': [
            (tf.layers.dense, {'units': 4096, 'activation': tf.nn.relu}),
            (tf.layers.dropout, {'rate': 0.5}),
            (tf.layers.dense, {'units': 4096, 'activation': tf.nn.relu}),
            (tf.layers.dropout, {'rate': 0.5}),
        ],
      Note that you don't need to provide "regularizer", "training",
      "data_format" and "axis" parameters since they will be
      automatically added. "axis" will be derived from "data_format" and will
      be ``1 if data_format == "channels_first" else 3``.

    * **data_format** (string) --- could be either "channels_first" or
      "channels_last". Defaults to "channels_first".
    """
    super(CNNEncoder, self).__init__(params, model, name, mode)

  def _encode(self, input_dict):
    regularizer = self.params.get('regularizer', None)
    data_format = self.params.get('data_format', 'channels_first')

    x = input_dict['source_tensors'][0]
    if data_format == 'channels_first':
      x = tf.transpose(x, [0, 3, 1, 2])

    for layer, layer_params in self.params['cnn_layers']:
      x = build_layer(x, layer, layer_params, data_format,
                      regularizer, self.mode == 'train')

    if data_format == 'channels_first':
      x = tf.transpose(x, [0, 2, 3, 1])

    fc_layers = self.params.get('fc_layers', [])

    # if fully connected layers exist, flattening the output and applying them
    if fc_layers:
      input_shape = x.get_shape().as_list()
      num_inputs = input_shape[1] * input_shape[2] * input_shape[3]
      x = tf.reshape(x, [-1, num_inputs])
      for layer, layer_params in fc_layers:
        x = build_layer(x, layer, layer_params, data_format, regularizer,
                        self.mode == 'train')
    else:
      # if there are no fully connected layers, doing average pooling
      x = tf.reduce_mean(x, [1, 2])

    return {'outputs': x}
