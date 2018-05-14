# Copyright (c) 2017 NVIDIA Corporation
"""Data Layer Classes"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import abc
import six
import tensorflow as tf
import copy
from open_seq2seq.utils.utils import check_params


@six.add_metaclass(abc.ABCMeta)
class DataLayer:
  """Abstract class from which all data layers must inherit.
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
    return {
      'mode': ['train', 'eval', 'infer'],
    }

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
      'batch_size': int,
      'shuffle': bool,
      'dtype': [tf.float32, tf.float16],
    }

  @abc.abstractmethod
  def __init__(self, params, model, num_workers=None, worker_id=None):
    """Data layer constructor.
    The TensorFlow graph should not be created here, but rather in the
    :meth:`self.build_graph() <build_graph>` method.
    Args:
      params (dict): parameters describing the data layer.
          All supported parameters are listed in :meth:`get_required_params`,
          :meth:`get_optional_params` functions.
      model (instance of a class derived from :class:`Model<models.model.Model>`):
          parent model that created this data layer.
          Could be None if no model access is required for the use case.
      num_workers (int): number of Horovod processes or None if Horovod is not used.
      worker_id (int): Horovod process id or None if Horovod is not used.
    Config parameters:
    * **shuffle** (bool) --- whether to shuffle dataset after an epoch.
      Typically will be True for train and False for inference and evaluation.
    * **dtype** --- data dtype. Could be either ``tf.float16`` or ``tf.float32``.
    """
    check_params(params, self.get_required_params(), self.get_optional_params())
    self._params = copy.deepcopy(params)
    self._model = model

    if 'dtype' not in self._params:
      if self._model:
        self._params['dtype'] = self._model.get_tf_dtype()
      else:
        self._params['dtype'] = tf.float32

    if 'shuffle' not in params:
      if self._params['mode'] == 'train':
        self._params['shuffle'] = True
      else:
        self._params['shuffle'] = False

    if self._params['mode'] != 'train' and self._params['shuffle']:
      raise ValueError("Shuffle should not be performed in eval or infer modes")

    # could be used for correct Horovod processing
    self._num_workers = num_workers
    self._worker_id = worker_id

  @property
  def params(self):
    """Parameters used to construct the data layer (dictionary)."""
    return self._params

  @abc.abstractmethod
  def build_graph(self):
    """Here all TensorFlow graph construction should happen."""
    pass

  @property
  @abc.abstractmethod
  def iterator(self):
    """Dataset iterator. Should be created by :meth:`build_graph`."""
    pass

  @property
  @abc.abstractmethod
  def input_tensors(self):
    """Returns input tensors that will be connected to the model graph.
    Should be created by :meth:`build_graph`.
    Returns:
      list: input tensors generated with
      :meth:`self.gen_input_tensors()<gen_input_tensors>`.
    """
    pass

  def get_size_in_samples(self):
    """Should return the dataset size in samples.
    That is, the number of objects in the dataset. This method is used to
    calculate a valid epoch size.
    Returns:
      int: dataset size in samples.
    """
    return None

  def get_size_in_batches(self):
    """Returns dataset size in batches.
    Returns:
      int: dataset size in batches.
    """
    size_in_samples = self.get_size_in_samples()
    if size_in_samples is None or 'batch_size' not in self.params:
      return None
    return self.get_size_in_samples() // self.params['batch_size']

  def split_data(self, data):
    if self.params['mode'] != 'train' and self._num_workers is not None:
      size = len(data)
      start = size // self._num_workers * self._worker_id
      if self._worker_id == self._num_workers - 1:
        end = size
      else:
        end = size // self._num_workers * (self._worker_id + 1)
      return data[start:end]
    else:
      return data
