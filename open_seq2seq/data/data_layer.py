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
      'batch_size': int,
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
      'shuffle': bool,
      'dtype': [tf.float32, tf.float16],
      'use_targets': bool,
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

  @abc.abstractmethod
  def gen_input_tensors(self):
    """This method should create and return input tensors that will be
    connected to the model computational graph. Note, that it is important to
    create *new* tensors here, since this method will be called multiple times
    for multi-GPU execution. For ``tf.data`` API you can usually call
    ``Iterator.get_next()`` multiple times.

    Returns:
       list: list of input tensors that could be placeholders or if using
       ``tf.data`` API, whatever is returned with ``Iterator.get_next()``.
    """
    pass

  @abc.abstractmethod
  def next_batch_feed_dict(self):
    """This method should return one data batch feed_dict.
    Basically, the output of this method will be included in the ``sess.run``
    call as the ``feed_dict`` parameter. If no ``feed_dict`` is required (as is
    the typical case with ``tf.data`` API), just return an empty dictionary.

    Returns:
      dict: feed_dict to be included in ``sess.run`` call.
    """
    pass

  @abc.abstractmethod
  def shuffle(self):
    """This method should implement data shuffle.
    It will be called after the end of each epoch. Note, that if shuffling is
    performed automatically (as is the typical case with ``tf.data`` API), this
    method can be empty.
    """
    pass

  @abc.abstractmethod
  def get_size_in_samples(self):
    """Should return the dataset size in samples.
    That is, the number of objects in the dataset. This method is used to
    calculate a valid epoch size.

    Returns:
      int: dataset size in samples.
    """
    pass

  def get_input_tensors(self):
    """Returns input tensors that will be connected to the model graph.
    Note: it is important **not to** overwrite this function for correct
    multi-GPU processing!

    Returns:
      list: input tensors generated with
      :meth:`self.gen_input_tensors()<gen_input_tensors>`.
    """
    if self._input_tensors is None:
      self._input_tensors = self.gen_input_tensors()
    return tuple(self._input_tensors)

  def get_size_in_batches(self):
    """Returns dataset size in batches.

    Returns:
      int: dataset size in batches.
    """
    return self.get_size_in_samples() // self.params['batch_size']

  def iterate_one_epoch(self, cross_over=False):
    """Generator that iterates through one epoch.

    Args:
      cross_over: whether last batch should take few elements from the next
          epoch if the size of dataset is not divisible by the batch size. If
          set to False, epoch will end ignoring few last elements.

    Yields:
       dict: feed_dict returned from
       :meth:`self.next_batch_feed_dict()<next_batch_feed_dict>` method.
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
    """Generator that goes through data set indefinitely.
    Will automatically perform shuffle after the end of each epoch by calling
    :meth:`self.shuffle()<shuffle>` method.

    Yields:
        dict: feed_dict returned from
       :meth:`self.next_batch_feed_dict()<next_batch_feed_dict>` method.
    """
    while True:
      for feed_dict in self.iterate_one_epoch():
        yield feed_dict
      if self.params['shuffle']:
        self.shuffle()


class MultiGPUWrapper(DataLayer):
  """Wrapper around :class:`DataLayer` class that enables multi-GPU execution.
  """
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
    """Wrapper constructor.

    Args:
      data_layer (instance of a class derived from :class:`DataLayer`): inner
          data_layer that will be "copied" to each GPU.
      num_gpus (int): number of GPUs used.
    """
    if not issubclass(type(data_layer), DataLayer):
      raise ValueError("data_layer has to be an instance "
                       "of a subclass of DataLayer class")
    super(MultiGPUWrapper, self).__init__(data_layer.params, data_layer._model)

    self._num_gpus = num_gpus
    self.params['batch_size'] *= self._num_gpus
    self._data_layer = data_layer

  def build_graph(self):
    """This function creates input tensors for all GPUs.
    It will first build graph for the inner data layer. Then, it will call
    :meth:`self._data_layer.gen_input_tensors()<gen_input_tensors>` num_gpus
    times and save the corresponding list as new ``self._input_tensors``
    parameter. This parameter will then be accessed by
    :meth:`self.get_input_tensors()<get_input_tensors>` method.
    """
    self._data_layer.build_graph()
    # making num_gpus copies of input tensors
    self._input_tensors = [
      self._data_layer.gen_input_tensors() for _ in range(self._num_gpus)
    ]
    # transposing, so that same type variables are in the same position
    self._input_tensors = list(zip(*self._input_tensors))

  def gen_input_tensors(self):
    """This function is empty since we directly fill ``self._input_tensors``
    which is used in :meth:`self.get_input_tensors()<get_input_tensors>` method.
    """
    pass

  def get_size_in_samples(self):
    """Redirects call to inner data layer :meth:`get_size_in_samples` method."""
    return self._data_layer.get_size_in_samples()

  def next_batch_feed_dict(self):
    """Correctly populates next batch feed_dict for multiple GPUs.
    This function will do the following things: for each GPU in a loop it will
    replace the inner data layer ``_input_tensors`` parameter with part of
    ``self._input_tensors`` corresponding to the current GPU. It will then
    call inner data layer :meth:`next_batch_feed_dict` method which should
    connect inner data layer ``_input_tensors`` with correct feed data. After
    all GPUs are covered it will return the complete feed dictionary.

    Note that this function will likely be rewritten in the future since it
    relies on the user using ``self._input_tensors`` to store the placeholders
    (which will be the case if they are accessed through
    :meth:`self.get_input_tensors()<get_input_tensors>` method),
    but this is not enforced in the API.

    Returns:
      dict: feed_dict to be included in ``sess.run`` call which has data for
      all GPUs.
    """
    feed_dict = {}
    for i in range(self._num_gpus):
      self._data_layer._input_tensors = tuple(
        tensors[i] for tensors in self._input_tensors
      )
      feed_dict.update(self._data_layer.next_batch_feed_dict())
    return feed_dict

  def shuffle(self):
    """Redirects call to inner data layer :meth:`shuffle` method."""
    self._data_layer.shuffle()
