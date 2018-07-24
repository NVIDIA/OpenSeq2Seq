# This code is heavily based on the code from TensorFlow official models
# https://github.com/tensorflow/models/tree/master/official/resnet

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import os

import numpy as np
import tensorflow as tf
from six.moves import range

from open_seq2seq.data.data_layer import DataLayer
from .imagenet_preprocessing import parse_record


class CifarDataLayer(DataLayer):
  _HEIGHT = 28
  _WIDTH = 28
  _NUM_CHANNELS = 3
  _DEFAULT_IMAGE_BYTES = 32 * 32 * 3
  # The record is the image plus a one-byte label
  _RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
  _NUM_CLASSES = 10
  _NUM_DATA_FILES = 5

  _NUM_IMAGES = {
      'train': 50000,
      'validation': 10000,
  }

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
        'image_size': int,
        'num_classes': int,
    })

  def __init__(self, params, model, num_workers, worker_id):
    super(CifarDataLayer, self).__init__(params, model,
                                         num_workers, worker_id)
    if self.params['mode'] == 'infer':
      raise ValueError('Inference is not supported on CifarDataLayer')

    if self.params['mode'] == 'train':
      filenames = [
          os.path.join(self.params['data_dir'], 'data_batch_{}.bin'.format(i))
          for i in range(1, self._NUM_DATA_FILES + 1)
      ]
    else:
      filenames = [os.path.join(self.params['data_dir'], 'test_batch.bin')]

    self.file_names = filenames
    self._train_size = 50000
    self._valid_size = 10000
    self._iterator = None
    self._input_tensors = None

  def preprocess_image(self, image, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
      # Resize the image to add four extra pixels on each side.
      image = tf.image.resize_image_with_crop_or_pad(
          image, self._HEIGHT + 8, self._WIDTH + 8
      )

      # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
      image = tf.random_crop(image, [self._HEIGHT, self._WIDTH,
                                     self._NUM_CHANNELS])

      # Randomly flip the image horizontally.
      image = tf.image.random_flip_left_right(image)

    else:
      image = tf.image.resize_image_with_crop_or_pad(
          image, self._HEIGHT, self._WIDTH
      )

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)

    return image

  def parse_record(self, raw_record, is_training, num_classes=10):
    """Parse CIFAR-10 image and label from a raw record."""
    # Convert bytes to a vector of uint8 that is record_bytes long.
    record_vector = tf.decode_raw(raw_record, tf.uint8)

    # The first byte represents the label, which we convert from uint8 to int32
    # and then to one-hot.
    label = tf.cast(record_vector[0], tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(record_vector[1:self._RECORD_BYTES],
                             [3, 32, 32])

    # Convert from [depth, height, width] to [height, width, depth], and cast as
    # float32.
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    image = self.preprocess_image(image, is_training)
    label = tf.one_hot(tf.reshape(label, shape=[]), num_classes)

    return image, label

  def build_graph(self):
    dataset = tf.data.FixedLengthRecordDataset(self.file_names,
                                               self._RECORD_BYTES)

    dataset = dataset.prefetch(buffer_size=self.params['batch_size'])
    if self.params['shuffle']:
      # shuffling images
      dataset = dataset.shuffle(buffer_size=self.params.get('shuffle_buffer',
                                                            1500))
    dataset = dataset.repeat()

    dataset = dataset.map(
        lambda value: self.parse_record(
            raw_record=value,
            is_training=self.params['mode'] == 'train',
        ),
        num_parallel_calls=self.params.get('num_parallel_calls', 16),
    )

    dataset = dataset.batch(self.params['batch_size'])
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    self._iterator = dataset.make_initializable_iterator()
    inputs, labels = self.iterator.get_next()
    if self.params['mode'] == 'train':
      tf.summary.image('augmented_images', inputs, max_outputs=1)
    self._input_tensors = {
        'source_tensors': [inputs],
        'target_tensors': [labels],
    }

  @property
  def input_tensors(self):
    return self._input_tensors

  @property
  def iterator(self):
    return self._iterator

  def get_size_in_samples(self):
    if self.params['mode'] == 'train':
      return self._train_size
    return len(np.arange(self._valid_size)[self._worker_id::self._num_workers])


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
        'image_size': int,
        'num_classes': int,
    })

  def __init__(self, params, model, num_workers, worker_id):
    super(ImagenetDataLayer, self).__init__(params, model,
                                            num_workers, worker_id)
    if self.params['mode'] == 'infer':
      raise ValueError('Inference is not supported on ImagenetDataLayer')

    if self.params['mode'] == 'train':
      filenames = [
          os.path.join(self.params['data_dir'],
                       'train-{:05d}-of-01024'.format(i))
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
        lambda value: parse_record(
            raw_record=value,
            is_training=self.params['mode'] == 'train',
            image_size=self.params.get('image_size', 224),
            num_classes=self.params.get('num_classes', 1000),
        ),
        num_parallel_calls=self.params.get('num_parallel_calls', 16),
    )

    dataset = dataset.batch(self.params['batch_size'])
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    self._iterator = dataset.make_initializable_iterator()
    inputs, labels = self.iterator.get_next()
    if self.params['mode'] == 'train':
      tf.summary.image('augmented_images', inputs, max_outputs=1)
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
    return self._valid_size
