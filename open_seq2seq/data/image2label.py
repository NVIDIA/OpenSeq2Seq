# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import sys
import os

from .data_layer import DataLayer

sys.path.insert(0, os.path.abspath("tensorflow-models"))
from official.resnet.imagenet_main import input_fn as imagenet_dataset


class ImagenetDataLayer(DataLayer):
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'is_training': bool,
      'data_dir': str,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{

    })

  def __init__(self, params, model, num_workers=None, worker_id=None):
    super(ImagenetDataLayer, self).__init__(params, model,
                                               num_workers, worker_id)
    self.iterator = None

  def build_graph(self):
    dataset = imagenet_dataset(
      is_training=self.params['is_training'],
      data_dir=self.params['data_dir'],
      batch_size=self.params['batch_size'],
      num_epochs=1000,  # need to set more than any reasonable epoch number
      num_parallel_calls=16,  # approximately number of CPU threads
      multi_gpu=False,  # TODO: is this correct?
    )
    self.iterator = dataset.make_one_shot_iterator()

  def gen_input_tensors(self):
    return self.iterator.get_next()

  def shuffle(self):
    """This method is empty, since shuffling is performed by ``tf.data``."""
    pass

  def get_size_in_samples(self):
    if self.params['is_training']:
      return 1281167
    else:
      return 50000

  def next_batch_feed_dict(self):
    """This method is empty, since ``tf.data`` does not need feed dictionary."""
    return {}
