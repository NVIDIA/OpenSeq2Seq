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
      'batch_size_per_gpu': int,
      'dtype': [tf.float32, tf.float16, 'mixed'],
    }

  def __init__(self, params, name="encoder", mode='train'):
    """Encoder constructor.
    Note that encoder constructors should not modify TensorFlow graph, all
    graph construction should happen in the :meth:`self._encode() <_encode>`
    method.

    Args:
      params (dict): parameters describing the encoder.
          All supported parameters are listed in :meth:`get_required_params`,
          :meth:`get_optional_params` functions.
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
    self._model = None  # will be populated in self.set_model() method

  def set_model(self, model):
    """Sets parent model to self._model attribute.
    Useful for intra-class communication, for example when decoder needs to
    access data layer property (e.g. vocabulary size).
    """
    self._model = model

  def encode(self, input_dict):
    """Wrapper around :meth:`self._encode() <_encode>` method.
    Here name, initializer and dtype are set in the variable scope and then
    :meth:`self._encode() <_encode>` method is called.

    Args:
      input_dict (dict): see :meth:`self._encode() <_encode>` docs.

    Returns:
      see :meth:`self._encode() <_encode>` docs.
    """
    if self._model is None:
      raise RuntimeError("Model attribute is not set. Make sure set_model() "
                         "method was called")

    if 'initializer' in self.params:
      init_dict = self.params.get('initializer_params', {})
      initializer = self.params['initializer'](**init_dict)
    else:
      initializer = None
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
    """This is the main function which should construct encoder graph.
    Typically, encoder will take raw input sequence as an input and
    produce some hidden representation as an output.

    Args:
      input_dict (dict): dictionary containing encoder inputs. This dict will
          typically have the following content::
            {
              "src_inputs": source_sequence,
              "src_lengths": src_length,
            }

    Returns:
      dict:
        dictionary of encoder outputs. Return all necessary outputs.
        Typically this will be just::
          {
            "encoder_output": encoder_output,
            "encoder_state": encoder_state,
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
