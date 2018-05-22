# This code is heavily based on the code from TensorFlow official models
# https://github.com/tensorflow/models/tree/master/official/resnet

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import os
import tensorflow as tf

from open_seq2seq.data.data_layer import DataLayer
from .imagenet_preprocessing import parse_record


class ImagenetDataLayer(DataLayer):
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'data_dir': str,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'num_parallel_calls': int,
      'shuffle_buffer': int,
    })

  def __init__(self, params, model, num_workers, worker_id):
    super(ImagenetDataLayer, self).__init__(params, model,
                                            num_workers, worker_id)
    if self.params['mode'] == 'infer':
      raise ValueError('Inference is not supported on ImagenetDataLayer')

    if self.params['mode'] == 'train':
      filenames = [
        os.path.join(self.params['data_dir'], 'train-{:05d}-of-01024'.format(i))
        for i in range(1024)  # number of training files
      ]
    else:
      filenames = [
        os.path.join(self.params['data_dir'],
                     'validation-{:05d}-of-00128'.format(i))
        for i in range(128)  # number of validation files
      ]

    self._train_size = 1281167
    self._valid_size = 0

    self.file_names = self.split_data(filenames)

    # TODO: rewrite this somehow?
    if self.params['mode'] != 'train':
      for file_name in self.file_names:
        for _ in tf.python_io.tf_record_iterator(file_name):
          self._valid_size += 1

    self._iterator = None
    self._input_tensors = None

  def build_graph(self):
    dataset = tf.data.Dataset.from_tensor_slices(self.file_names)

    if self.params['shuffle']:
      # shuffling input files
      dataset = dataset.shuffle(buffer_size=1024)

    # convert to individual records
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    dataset = dataset.prefetch(buffer_size=self.params['batch_size'])
    if self.params['shuffle']:
      # shuffling images
      dataset = dataset.shuffle(buffer_size=self.params.get('shuffle_buffer',
                                                            1500))
    dataset = dataset.repeat()

    dataset = dataset.map(
      lambda value: parse_record(value, self.params['mode'] == 'train'),
      num_parallel_calls=self.params.get('num_parallel_calls', 16),
    )

    dataset = dataset.batch(self.params['batch_size'])
    dataset = dataset.prefetch(1)

    self._iterator = dataset.make_initializable_iterator()
    inputs, labels = self.iterator.get_next()
    self._input_tensors = {
      'source_tensors': [inputs],
      'target_tensors': [labels],
    }

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

  @property
  def input_tensors(self):
    return self._input_tensors

  @property
  def iterator(self):
    return self._iterator

  def get_size_in_samples(self):
    if self.params['mode'] == 'train':
      return self._train_size
    else:
      return self._valid_size
