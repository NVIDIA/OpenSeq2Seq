# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import abc
import six
import copy
import tensorflow as tf

from open_seq2seq.utils.utils import check_params


@six.add_metaclass(abc.ABCMeta)
class Loss:
  """Abstract class from which all losses must inherit.
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
      'batch_size_per_gpu': int,
      'dtype': [tf.float16, tf.float32, "mixed"],
    }

  def __init__(self, params, name="loss"):
    """Loss constructor.
    Note that loss constructors should not modify TensorFlow graph, all
    graph construction should happen in the
    :meth:`self._compute_loss() <_compute_loss>` method.

    Args:
      params (dict): parameters describing the loss.
          All supported parameters are listed in :meth:`get_required_params`,
          :meth:`get_optional_params` functions.
      name (str): name for loss variable scope.
    """
    check_params(params, self.get_required_params(), self.get_optional_params())
    self._params = copy.deepcopy(params)
    if 'dtype' not in params:
      self.params['dtype'] = tf.float32
    if self.params['dtype'] == 'mixed':
      self.params['dtype'] = tf.float16

    self._name = name
    self._model = None  # will be populated in self.set_model() method

  def set_model(self, model):
    """Sets parent model to self._model attribute.
    Useful for intra-class communication, for example when decoder needs to
    access data layer property (e.g. vocabulary size).
    """
    self._model = model

  def compute_loss(self, input_dict):
    """Wrapper around :meth:`self._compute_loss() <_compute_loss>` method.
    Here name and dtype are set in the variable scope and then
    :meth:`self._compute_loss() <_compute_loss>` method is called.

    Args:
      input_dict (dict): see :meth:`self._compute_loss() <_compute_loss>` docs.

    Returns:
      see :meth:`self._compute_loss() <_compute_loss>` docs.
    """
    if self._model is None:
      raise RuntimeError("Model attribute is not set. Make sure set_model() "
                         "method was called")

    with tf.variable_scope(self._name, dtype=self.params['dtype']):
      return self._compute_loss(self._cast_types(input_dict))

  def _cast_types(self, input_dict):
    """This function performs automatic cast of all inputs to the loss dtype.

    Args:
      input_dict (dict): dictionary passed to
          :meth:`self._compute_loss() <_compute_loss>` method.

    Returns:
      dict: same as input_dict, but with all Tensors cast to the loss dtype.
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
  def _compute_loss(self, input_dict):
    """This is the main function which should construct loss graph.
    Typically, loss will take decoder-produced logits as an input and
    return a singleton loss tensor.

    Args:
      input_dict (dict): dictionary containing loss inputs. This dict will
          typically have the following content::
            {
              "logits": decoder_logits,
              "tgt_sequence": target_sequence,
              "tgt_length": target_length,
            }

    Returns:
      singleton loss tensor. This tensor will be computed independently
      for each GPU batch and then averaged
      (``reduce_mean``) over the number of GPUs (or Horovod workers).
    """
    pass

  @property
  def params(self):
    """Parameters used to construct the loss (dictionary)."""
    return self._params

  @property
  def name(self):
    """Loss name."""
    return self._name
