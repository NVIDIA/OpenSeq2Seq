# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
import copy
from .encoder import Encoder
from open_seq2seq.utils.utils import deco_print


def build_layer(inputs, layer, layer_params, data_format, regularizer):
  layer_built = False

  for reg_name in ['regularizer', 'kernel_regularizer',
                   'gamma_regularizer', None]:
    if layer_built:
      break
    for try_data_format in [True, False]:
      cur_params = copy.deepcopy(layer_params)
      if try_data_format:
        cur_params.update({'data_format': data_format})
      if reg_name is not None:
        cur_params.update({reg_name: regularizer})
      try:
        outputs = layer(inputs, **cur_params)
        layer_built = True
        break
      except TypeError as e:
        if "got an unexpected keyword argument '{}'".format(reg_name) in e.__str__():
          continue
        if "got an unexpected keyword argument 'data_format'" in e.__str__():
          continue
        raise

  if not layer_built:
    cur_params = copy.deepcopy(layer_params)
    outputs = layer(inputs, **cur_params)

  if hasattr(layer, '_tf_api_names'):
    layer_name = layer._tf_api_names[0]
  else:
    layer_name = layer
  deco_print("Building layer: {}(inputs, {})".format(
    layer_name,
    ", ".join("{}={}".format(key, value) for key, value in cur_params.items())
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
      x = build_layer(x, layer, layer_params, data_format, regularizer)

    if data_format == 'channels_first':
      x = tf.transpose(x, [0, 2, 3, 1])
    input_shape = x.get_shape().as_list()
    num_inputs = input_shape[1] * input_shape[2] * input_shape[3]
    x = tf.reshape(x, [-1, num_inputs])

    for layer, layer_params in self.params.get('fc_layers', []):
      x = build_layer(x, layer, layer_params, data_format, regularizer)

    return {'outputs': x}
