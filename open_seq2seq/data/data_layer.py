# Copyright (c) 2017 NVIDIA Corporation
"""Data Layer Classes"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

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
      'batch_size': int,
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
      num_workers (int): number of Horovod processes or shards.
      worker_id (int): Horovod process id or shard id.
    """
    check_params(params, self.get_required_params(), self.get_optional_params())
    self._params = copy.deepcopy(params)
    self._model = model
    self._iterator = None

    if 'dtype' not in self._params:
      if self._model:
        self._params['dtype'] = self._model.get_tf_dtype()
      else:
        self._params['dtype'] = tf.float32

    self._num_workers = num_workers
    self._worker_id = worker_id

  @property
  def params(self):
    """Parameters used to construct the data layer (dictionary)."""
    return self._params

  @abc.abstractmethod
  def build_graph(self):
    """Here all TensorFlow graph construction should happen.
    Inside this function self._iterator should be created"""
    pass

  @abc.abstractmethod
  def get_iterator(self):
    """This method return initializable TF.data iterator

    Returns:
       An initializable iterator of type tf.data.Iterator
    """
    return self._iterator

  # Oleksii: I think we should get rid of this method too
  def gen_input_tensors(self):
    """This method should create and return input tensors that will be
    connected to the model computational graph.

    Returns:
       list: Whatever is returned with ``Iterator.get_next()``.
    """
    return self._iterator.get_next()

# Oleksii: I think we are getting rid of this class at least in its current form
# See https://stackoverflow.com/questions/46965098/how-does-one-move-data-to-multiple-gpu-towers-using-tensorflows-dataset-api
# for approaches with multi-GPU and tf.data API
# Personally I like their option 3 and using dataset.shard() method as it fits nicely to Horovod's num_workers and worker_id
# scenario
# d = d.shard(FLAGS.num_workers, FLAGS.worker_index) see https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shard
