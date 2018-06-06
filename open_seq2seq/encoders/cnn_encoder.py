# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
import copy

try:
    from inspect import signature
except ImportError:
    from funcsigs import signature

from .encoder import Encoder
from open_seq2seq.utils.utils import deco_print


def build_layer(inputs, layer, layer_params, data_format,
                regularizer, training, verbose=True):
  layer_params_cp = copy.deepcopy(layer_params)
  for reg_name in ['regularizer', 'kernel_regularizer', 'gamma_regularizer']:
    if reg_name not in layer_params_cp and \
       reg_name in signature(layer).parameters:
      layer_params_cp.update({reg_name: regularizer})

  if 'data_format' not in layer_params_cp and \
     'data_format' in signature(layer).parameters:
    layer_params_cp.update({'data_format': data_format})

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
