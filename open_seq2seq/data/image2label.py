# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import sys
import os
import tensorflow as tf

from .data_layer import DataLayer

sys.path.insert(0, os.path.abspath("tensorflow-models"))
from official.resnet import resnet_run_loop
from official.resnet.imagenet_main import get_filenames, _NUM_TRAIN_FILES, \
                                          _NUM_IMAGES, _SHUFFLE_BUFFER, \
                                          parse_record


class ImagenetDataLayer(DataLayer):
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
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
    self.valid_size = 0

  def build_graph(self):
    file_names = self.split_data(get_filenames(self.params['mode'] == 'train',
                                               self.params['data_dir']))
    if self.params['mode'] != 'train':
      for file_name in file_names:
        for _ in tf.python_io.tf_record_iterator(file_name):
          self.valid_size += 1

    dataset = tf.data.Dataset.from_tensor_slices(file_names)

    if self.params['mode'] == 'train':
      # Shuffle the input files
      dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

    if self.params['mode'] == 'train':
      num_images = _NUM_IMAGES['train']
    else:
      num_images = self.valid_size

    # Convert to individual records
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    dataset = resnet_run_loop.process_record_dataset(
      dataset=dataset,
      is_training=self.params['mode'] == 'train',
      batch_size=self.params['batch_size'],
      shuffle_buffer=_SHUFFLE_BUFFER,
      parse_record_fn=parse_record,
      num_epochs=1000,
      num_parallel_calls=16,
      examples_per_epoch=num_images,
      multi_gpu=False,
    )

    self.iterator = dataset.make_one_shot_iterator()

  def gen_input_tensors(self):
    return self.iterator.get_next()

  def shuffle(self):
    """This method is empty, since shuffling is performed by ``tf.data``."""
    pass

  def get_size_in_samples(self):
    if self.params['mode'] == 'train':
      return _NUM_IMAGES['train']
    else:
      return self.valid_size

  def next_batch_feed_dict(self):
    """This method is empty, since ``tf.data`` does not need feed dictionary."""
    return {}
