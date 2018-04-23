# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import abc
import six
import tensorflow as tf
import copy
from open_seq2seq.utils.utils import check_params
from open_seq2seq.optimizers.mp_wrapper import mp_regularizer_wrapper


@six.add_metaclass(abc.ABCMeta)
class Decoder:
  """Abstract class from which all decoders must inherit.
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
      'batch_size': int,
      'dtype': [tf.float32, tf.float16, 'mixed'],
    }

  def __init__(self, params, model, name="decoder", mode='train'):
    """Decoder constructor.
    Note that decoder constructors should not modify TensorFlow graph, all
    graph construction should happen in the :meth:`self._decode() <_decode>`
    method.

    Args:
      params (dict): parameters describing the decoder.
          All supported parameters are listed in :meth:`get_required_params`,
          :meth:`get_optional_params` functions.
      name (str): name for decoder variable scope.
      mode (str): mode decoder is going to be run in.
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
      self._params['dtype'] = self._model.params['dtype']

    if 'regularizer' not in self._params:
      if 'regularizer' in self._model.params:
        self._params['regularizer'] = self._model.params['regularizer']
        self._params['regularizer_params'] = self._model.params['regularizer_params']

    if 'regularizer' in self._params:
      init_dict = self._params.get('regularizer_params', {})
      self._params['regularizer'] = self._params['regularizer'](**init_dict)
      if self._params['dtype'] == 'mixed':
        self._params['regularizer'] = mp_regularizer_wrapper(
          self._params['regularizer'],
        )

    if self._params['dtype'] == 'mixed':
      self._params['dtype'] = tf.float16

    self._name = name
    self._mode = mode

  def decode(self, input_dict):
    """Wrapper around :meth:`self._decode() <_decode>` method.
    Here name, initializer and dtype are set in the variable scope and then
    :meth:`self._decode() <_decode>` method is called.

    Args:
      input_dict (dict): see :meth:`self._decode() <_decode>` docs.

    Returns:
      see :meth:`self._decode() <_decode>` docs.
    """
    if 'initializer' in self.params:
      init_dict = self.params.get('initializer_params', {})
      initializer = self.params['initializer'](**init_dict)
    else:
      initializer = None

    with tf.variable_scope(self._name, initializer=initializer,
                           dtype=self.params['dtype']):
      return self._decode(self._cast_types(input_dict))

  def _cast_types(self, input_dict):
    """This function performs automatic cast of all inputs to decoder dtype.

    Args:
      input_dict (dict): dictionary passed to :meth:`self._decode() <_decode>`
          method.

    Returns:
      dict: same as input_dict, but with all Tensors cast to decoder dtype.
    """
    def _cast_dict(dict_to_cast):
      cast_dict = {}
      for key, value in dict_to_cast.items():
        if isinstance(value, tf.Tensor):
          if value.dtype == tf.float16 or value.dtype == tf.float32:
            if value.dtype != self.params['dtype']:
              cast_dict[key] = tf.cast(value, self.params['dtype'])
              continue
        cast_dict[key] = value
      return cast_dict

    # TODO: do we need to add some recursion to parse all nested dicts?
    cast_input_dict = _cast_dict(input_dict)
    cast_input_dict['encoder_output'] = _cast_dict(input_dict['encoder_output'])
    return cast_input_dict

  @abc.abstractmethod
  def _decode(self, input_dict):
    """This is the main function which should construct decoder graph.
    Typically, decoder will take hidden representation from encoder as an input
    and produce some output sequence as an output.

    Args:
      input_dict (dict): dictionary containing decoder inputs. This dict will
          typically have the following content::
            {
              "encoder_output": encoder_output,
              "tgt_inputs": target_sequence,
              "tgt_lengths": target_lengths,
            }

    Returns:
      dict:
        dictionary of decoder outputs. Typically this will be just::
          {
            "decoder_output": decoder_logits,  # what will be passed to Loss
            "decoder_samples": decoder_samples,  # actual decoded sequence, e.g.
                                                 # characters instead of logits
          }
    """
    pass

  @property
  def params(self):
    """Parameters used to construct the decoder (dictionary)"""
    return self._params

  @property
  def mode(self):
    """Mode decoder is run in."""
    return self._mode

  @property
  def name(self):
    """Decoder name."""
    return self._name
