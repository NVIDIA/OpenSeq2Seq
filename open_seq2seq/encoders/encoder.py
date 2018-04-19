# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import abc
import six
import tensorflow as tf
import copy
from open_seq2seq.utils.utils import check_params
from open_seq2seq.optimizers.mp_wrapper import mp_regularizer_wrapper


@six.add_metaclass(abc.ABCMeta)
class Encoder:
  """Abstract class from which all encoders must inherit.
  """
  @staticmethod
  def get_required_params():
    return {}

  @staticmethod
  def get_optional_params():
    return {
      'regularizer': None,  # any valid TensorFlow regularizer
      'regularizer_params': dict,
      'initializer': None,  # any valid TensorFlow initializer
      'initializer_params': dict,
      'batch_size_per_gpu': int,
      'dtype': [tf.float32, tf.float16, 'mixed'],
    }

  def __init__(self, params, name="encoder", mode='train'):
    """
    Initializes Encoder. Encoder constructors should not modify TF graph
    :param params: dictionary of encoder parameters
    """
    check_params(params, self.get_required_params(), self.get_optional_params())
    self._params = copy.deepcopy(params)
    if 'dtype' not in params:
      self.params['dtype'] = tf.float32

    if 'regularizer' in self.params:
      init_dict = self.params.get('regularizer_params', {})
      self.params['regularizer'] = self.params['regularizer'](**init_dict)
      if self.params['dtype'] == 'mixed':
        self.params['regularizer'] = mp_regularizer_wrapper(
          self.params['regularizer'],
        )

    if self.params['dtype'] == 'mixed':
      self.params['dtype'] = tf.float16

    self._name = name
    self._mode = mode

  def encode(self, input_dict):
    if 'initializer' in self.params:
      init_dict = self.params.get('initializer_params', {})
      initializer = self.params['initializer'](**init_dict)
    else:
      initializer = None
    with tf.variable_scope(self._name, initializer=initializer,
                           dtype=self.params['dtype']):
      return self._encode(self._cast_types(input_dict))

  def _cast_types(self, input_dict):
    cast_input_dict = {}
    for key, value in input_dict.items():
      if isinstance(value, tf.Tensor):
        if value.dtype == tf.float16 or value.dtype == tf.float32:
          if value.dtype != self.params['dtype']:
            cast_input_dict[key] = tf.cast(value, self.params['dtype'])
            continue
      cast_input_dict[key] = value
    return cast_input_dict

  @abc.abstractmethod
  def _encode(self, input_dict):
    """
    Encodes encoder input sequence. This will add to TF graph
    Typically, encoder will take raw data as an input and will
    produce representation
    :param input_dict: dictionary of encoder inputs
    For example (but may differ):
    encoder_input= { "src_inputs" : source_sequence,
                     "src_lengths" : src_length }

    :return: dictionary of encoder outputs
    For example (but may differ):
    input_dict = {"encoder_outputs" : encoder_outputs,
                      "encoder_state" : encoder_state,
                      "src_lengths" : src_lengths}
    """
    pass

  @property
  def params(self):
    """Parameters used to construct the encoder"""
    return self._params
