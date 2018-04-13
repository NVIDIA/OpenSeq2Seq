# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import abc
import six
import copy
from open_seq2seq.utils import check_params


@six.add_metaclass(abc.ABCMeta)
class Loss:
  """Abstract class from which all losses must inherit.
  """
  @staticmethod
  def get_required_params():
    return {}

  @staticmethod
  def get_optional_params():
    return {
      'batch_size_per_gpu': int,
    }

  def __init__(self, params):
    """
    Constructor
    :param params: Python dictionary with parameters to initialize loss class
    """
    check_params(params, self.get_required_params(), self.get_optional_params())
    self._params = copy.deepcopy(params)

  def compute_loss(self, input_dict):
    """
    Computes loss.
    WARNING: The expectation is that this loss will be averaged (reduce_mean)
    over the number of GPUs (or Horovod workers)
    :param input_dict: dictionary with inputs to the loss function
    :return: singleton loss tensor
    """
  @property
  def params(self):
    return self._params
