# Copyright (c) 2017 NVIDIA Corporation
"""Data Layer Classes"""
from __future__ import absolute_import, division, print_function
import abc
import six
import tensorflow as tf
import copy
from open_seq2seq.utils.utils import check_params


@six.add_metaclass(abc.ABCMeta)
class DataLayer:
  """Abstract class that specifies data access operations
  """
  @staticmethod
  def get_required_params():
    return {
      'batch_size': int,
    }

  @staticmethod
  def get_optional_params():
    return {
      'shuffle': bool,
      'dtype': [tf.float32, tf.float16],
      'use_targets': bool,
    }

  @abc.abstractmethod
  def __init__(self, params, model):
    """
    Initialize data layer
    :param params: Python dictionary with options,
    specifying mini-batch shapes, padding, etc.
    """
    check_params(params, self.get_required_params(), self.get_optional_params())
    self._params = copy.deepcopy(params)
    self._model = model

    if 'dtype' not in self._params:
      self._params['dtype'] = self._model.get_tf_dtype()

    if 'use_targets' not in params:
      self._params['use_targets'] = True

    if 'shuffle' not in params:
      if self._params['use_targets']:
        self._params['shuffle'] = True
      else:
        self._params['shuffle'] = False

    if self._params['use_targets'] is False and self._params['shuffle']:
      raise ValueError("Shuffle should not be performed in inference mode")

    self._input_tensors = None

  @property
  def params(self):
    return self._params

  @abc.abstractmethod
  def gen_input_tensors(self):
    """
    Creates and returns input tensors that should be connected to the
    model computational graph.

    :return: list of input tensors: that could be placeholders or if using
             tf.data API, whatever is returned with Iterator.get_next()
    """
    pass

  @abc.abstractmethod
  def next_batch_feed_dict(self):
    """Should return one batch: something that can be fed into feed_dict"""
    pass

  @abc.abstractmethod
  def shuffle(self):
    """
    Shuffles the data.
    """
    pass

  @abc.abstractmethod
  def get_size_in_samples(self):
    """
    :return: dataset size in samples, i.e. number of training objects in dataset
    """
    pass

  def get_input_tensors(self):
    """
    Returns input tensors that should be connected to the
    model computational graph.

    :return: list of input tensors: that could be placeholders or if using
             tf.data API, whatever is returned with Iterator.get_next()
    """
    if self._input_tensors is None:
      self._input_tensors = self.gen_input_tensors()
    return tuple(self._input_tensors)

  def get_size_in_batches(self):
    """
    :return: dataset size in batches
    """
    return self.get_size_in_samples() // self.params['batch_size']

  def iterate_one_epoch(self, cross_over=False):
    """
    Goes through the data one time.
    :param cross_over: whether last batch should take few elements from the next
                       epoch if the size of dataset is not divisible by
                       the batch size
    :return: yields feed_dict that should populate the data into
             tensors returned by get_input_tensors function
    """
    if self.get_size_in_batches() == 0:
      raise ValueError(
        "Batch size is bigger than dataset size: {} > {}".format(
          self.params['batch_size'], self.get_size_in_samples()
        )
      )
    for _ in range(self.get_size_in_batches()):
      yield self.next_batch_feed_dict()
    if cross_over:
      if self.get_size_in_samples() % self.params['batch_size'] != 0:
        yield self.next_batch_feed_dict()

  def iterate_forever(self):
    """
    Goes through data set indefinitely
    :return: yields feed_dict that should populate the data into 
             tensors returned by get_input_tensors function.
             For tf.data API feed_dict will usually be empty
    """
    while True:
      for feed_dict in self.iterate_one_epoch():
        yield feed_dict
      if self.params['shuffle']:
        self.shuffle()


class MultiGPUWrapper(DataLayer):
  @staticmethod
  def get_required_params():
    # this disables the check since it was already done
    # inside the inner data_layer
    return None

  @staticmethod
  def get_optional_params():
    # this disables the check since it was already done
    # inside the inner data_layer
    return None

  def __init__(self, data_layer, num_gpus):
    if not issubclass(type(data_layer), DataLayer):
      raise ValueError("data_layer has to be an instance "
                       "of a subclass of DataLayer class")
    super(MultiGPUWrapper, self).__init__(data_layer.params, data_layer._model)

    self._num_gpus = num_gpus
    self.params['batch_size'] *= self._num_gpus
    self._data_layer = data_layer

    # making num_gpus copies of input tensors
    self._input_tensors = [
      self._data_layer.gen_input_tensors() for _ in range(self._num_gpus)
    ]
    # transposing, so that same type variables are in the same position
    self._input_tensors = list(zip(*self._input_tensors))

  def gen_input_tensors(self):
    # this function is unnecessary since we directly fill self._input_tensors
    pass

  def get_size_in_samples(self):
    return self._data_layer.get_size_in_samples()

  def next_batch_feed_dict(self):
    feed_dict = {}
    for i in range(self._num_gpus):
      self._data_layer._input_tensors = tuple(
        tensors[i] for tensors in self._input_tensors
      )
      feed_dict.update(self._data_layer.next_batch_feed_dict())
    return feed_dict

  def shuffle(self):
    self._data_layer.shuffle()
