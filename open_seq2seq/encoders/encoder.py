# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import abc
import copy

import six
import tensorflow as tf

from open_seq2seq.optimizers.mp_wrapper import mp_regularizer_wrapper
from open_seq2seq.utils.utils import check_params, cast_types


@six.add_metaclass(abc.ABCMeta)
class Encoder:
  """Abstract class from which all encoders must inherit.
  """
  @staticmethod
  def get_required_params():
    """Static method with description of required parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **have to** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return {}

  @staticmethod
  def get_optional_params():
    """Static method with description of optional parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **can** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return {
        'regularizer': None,  # any valid TensorFlow regularizer
        'regularizer_params': dict,
        'initializer': None,  # any valid TensorFlow initializer
        'initializer_params': dict,
        'dtype': [tf.float32, tf.float16, 'mixed'],
    }

  def __init__(self, params, model, name="encoder", mode='train'):
    """Encoder constructor.
    Note that encoder constructors should not modify TensorFlow graph, all
    graph construction should happen in the :meth:`self._encode() <_encode>`
    method.

    Args:
      params (dict): parameters describing the encoder.
          All supported parameters are listed in :meth:`get_required_params`,
          :meth:`get_optional_params` functions.
      model (instance of a class derived from :class:`Model<models.model.Model>`):
          parent model that created this encoder.
          Could be None if no model access is required for the use case.
      name (str): name for encoder variable scope.
      mode (str): mode encoder is going to be run in.
          Could be "train", "eval" or "infer".

    Config parameters:

    * **initializer** --- any valid TensorFlow initializer. If no initializer
      is provided, model initializer will be used.
    * **initializer_params** (dict) --- dictionary that will be passed to
      initializer ``__init__`` method.
    * **regularizer** --- and valid TensorFlow regularizer. If no regularizer
      is provided, model regularizer will be used.
    * **regularizer_params** (dict) --- dictionary that will be passed to
      regularizer ``__init__`` method.
    * **dtype** --- model dtype. Could be either ``tf.float16``, ``tf.float32``
      or "mixed". For details see
      :ref:`mixed precision training <mixed_precision>` section in docs. If no
      dtype is provided, model dtype will be used.
    """
    check_params(params, self.get_required_params(), self.get_optional_params())
    self._params = copy.deepcopy(params)
    self._model = model

    if 'dtype' not in self._params:
      if self._model:
        self._params['dtype'] = self._model.params['dtype']
      else:
        self._params['dtype'] = tf.float32

    self._name = name
    self._mode = mode
    self._compiled = False

  def encode(self, input_dict):
    """Wrapper around :meth:`self._encode() <_encode>` method.
    Here name, initializer and dtype are set in the variable scope and then
    :meth:`self._encode() <_encode>` method is called.

    Args:
      input_dict (dict): see :meth:`self._encode() <_encode>` docs.

    Returns:
      see :meth:`self._encode() <_encode>` docs.
    """
    if not self._compiled:
      if 'regularizer' not in self._params:
        if self._model and 'regularizer' in self._model.params:
          self._params['regularizer'] = copy.deepcopy(
              self._model.params['regularizer']
          )
          self._params['regularizer_params'] = copy.deepcopy(
              self._model.params['regularizer_params']
          )

      if 'regularizer' in self._params:
        init_dict = self._params.get('regularizer_params', {})
        self._params['regularizer'] = self._params['regularizer'](**init_dict)
        if self._params['dtype'] == 'mixed':
          self._params['regularizer'] = mp_regularizer_wrapper(
              self._params['regularizer'],
          )

      if self._params['dtype'] == 'mixed':
        self._params['dtype'] = tf.float16

    if 'initializer' in self.params:
      init_dict = self.params.get('initializer_params', {})
      initializer = self.params['initializer'](**init_dict)
    else:
      initializer = None

    self._compiled = True

    with tf.variable_scope(self._name, initializer=initializer,
                           dtype=self.params['dtype']):
      return self._encode(self._cast_types(input_dict))

  def _cast_types(self, input_dict):
    """This function performs automatic cast of all inputs to encoder dtype.

    Args:
      input_dict (dict): dictionary passed to :meth:`self._encode() <_encode>`
          method.

    Returns:
      dict: same as input_dict, but with all Tensors cast to encoder dtype.
    """
    return cast_types(input_dict, self.params['dtype'])

  @abc.abstractmethod
  def _encode(self, input_dict):
    """This is the main function which should construct encoder graph.
    Typically, encoder will take raw input sequence as an input and
    produce some hidden representation as an output.

    Args:
      input_dict (dict): dictionary containing encoder inputs.
          If the encoder is used with :class:`models.encoder_decoder` class,
          ``input_dict`` will have the following content::
            {
              "source_tensors": data_layer.input_tensors['source_tensors']
            }

    Returns:
      dict:
        dictionary of encoder outputs. Return all necessary outputs.
        Typically this will be just::
          {
            "outputs": outputs,
            "state": state,
          }
    """
    pass

  @property
  def params(self):
    """Parameters used to construct the encoder (dictionary)."""
    return self._params

  @property
  def mode(self):
    """Mode encoder is run in."""
    return self._mode

  @property
  def name(self):
    """Encoder name."""
    return self._name
