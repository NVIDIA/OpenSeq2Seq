# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Original work Copyright (c) 2018 Mozilla Corporation
# Modified work Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
import six
import numpy as np
import tensorflow as tf
import pandas as pd
import codecs
import os
from six.moves import range

from .data_layer import DataLayer
from .speech_utils import get_speech_features_from_file


class Alphabet(object):
  def __init__(self, config_file):
    self._label_to_str = []
    self._str_to_label = {}
    self._size = 0
    with codecs.open(config_file, 'r', 'utf-8') as fin:
      for line in fin:
        if line[0:2] == '\\#':
          line = '#\n'
        elif line[0] == '#':
          continue
        self._label_to_str += line[:-1]  # remove the line ending
        self._str_to_label[line[:-1]] = self._size
        self._size += 1

  def string_from_label(self, label):
    return self._label_to_str[label]

  def label_from_string(self, string):
    return self._str_to_label[string]

  def size(self):
    return self._size


def text_to_char_array(original, alphabet):
  """
  Given a Python string ``original``, remove unsupported characters,
  map characters to integers and return a numpy array representing
  the processed string.
  """
  return np.asarray([alphabet.label_from_string(c) for c in original])


class Speech2TextDataLayer(DataLayer):
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'num_audio_features': int,
      'input_type': ['spectrogram', 'mfcc'],
      'alphabet_config_path': str,
      'dataset_path': list,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'augmentation': dict,
    })

  def __init__(self, params, model):
    """
    Required params:
      batch_size_per_gpu
      alphabet_config_path
      dataset_path
      num_audio_features
      input_type
    Example:
      params = {
        'batch_size': 16,
        'alphabet_config_path': 'data/alphabet.txt',
        'dataset_path': ['/raid/data/speech/WSJ/wsj-train-128.csv'],
        'num_audio_features': 161,
        'input_type': 'spectrogram',
      }
    """
    super(Speech2TextDataLayer, self).__init__(params, model)

    self.params['alphabet'] = Alphabet(
      os.path.abspath(params['alphabet_config_path'])
    )
    self.params['tgt_vocab_size'] = self.params['alphabet'].size() + 1
    
    self._index = -1
    self._files = None
    for csv in params['dataset_path']:
      files = pd.read_csv(csv, encoding='utf-8')
      if self._files is None:
        self._files = files
      else:
        self._files = self._files.append(files)
    if self.params['use_targets']:
      cols = ["wav_filename", "transcript"]
    else:
      cols = "wav_filename"
    self._files = self._files.loc[:, cols].values
    self.params['files'] = self._files

    if self.params['shuffle']:
      self.shuffle()

    self._size = self.get_size_in_samples()

  def shuffle(self):
    self._files = np.random.permutation(self._files)

  def gen_input_tensors(self):
    x = tf.placeholder(
      self.params['dtype'],
      [self.params['batch_size'], None,
       self.params['num_audio_features']],
      name="ph_x",
    )
    x_length = tf.placeholder(
      tf.int32,
      [self.params['batch_size']],
      name="ph_xlen",
    )
    if self.params['use_targets']:
      y = tf.placeholder(
        tf.int32,
        [self.params['batch_size'], None],
        name="ph_y",
      )
      y_length = tf.placeholder(
        tf.int32,
        [self.params['batch_size']],
        name="ph_ylen",
      )
      return [x, x_length, y, y_length]
    else:
      return [x, x_length]

  def get_size_in_samples(self):
    return len(self._files)

  def get_one_sample(self):
    self._index += 1
    if self._index == self._size:
      self._index = 0

    if self.params['use_targets']:
      wav_file, transcript = self._files[self._index]
      target = text_to_char_array(transcript, self.params['alphabet'])
    else:
      wav_file = self._files[self._index]

    source = get_speech_features_from_file(
      wav_file, self.params['num_audio_features'],
      features_type=self.params['input_type'],
      augmentation=self.params.get('augmentation', None),
    )
    if self.params['use_targets']:
      return source, target
    else:
      return source

  def next_batch(self):
    sources = []
    sources_len = np.empty(self.params['batch_size'], dtype=np.int)
    max_length_sc = 0

    if self.params['use_targets']:
      targets = []
      targets_len = np.empty(self.params['batch_size'], dtype=np.int)
      max_length_tg = 0

    for i in range(self.params['batch_size']):
      if self.params['use_targets']:
        source, target = self.get_one_sample()
      else:
        source = self.get_one_sample()
      sources.append(source)
      sources_len[i] = source.shape[0]
      max_length_sc = max(max_length_sc, source.shape[0])

      if self.params['use_targets']:
        targets.append(target)
        targets_len[i] = target.shape[0]
        max_length_tg = max(max_length_tg, target.shape[0])

    for i in range(self.params['batch_size']):
      sources[i] = np.pad(
        sources[i],
        [(0, max_length_sc - sources[i].shape[0]), (0, 0)],
        mode='constant',
      )
      if self.params['use_targets']:
        targets[i] = np.pad(
          targets[i],
          (0, max_length_tg - targets[i].shape[0]),
          mode='constant',
        )
    if self.params['use_targets']:
      return np.array(sources), sources_len, np.array(targets), targets_len
    else:
      return np.array(sources), sources_len

  def next_batch_feed_dict(self):
    return {self.get_input_tensors(): self.next_batch()}


class Speech2TextRandomDataLayer(DataLayer):
  """This class should be used for performance profiling only."""
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'num_audio_features': int,
      'input_type': ['spectrogram', 'mfcc'],
      'alphabet_config_path': str,
      'dataset_path': list,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'augmentation': dict,
    })

  def __init__(self, params, model):
    """
    Random data for speech check
    """
    super(Speech2TextRandomDataLayer, self).__init__(params, model)
    self.random_data = None
    self.params['alphabet'] = Alphabet(
      os.path.abspath(params['alphabet_config_path'])
    )
    self.params['tgt_vocab_size'] = self.params['alphabet'].size() + 1

  def shuffle(self):
    pass

  def gen_input_tensors(self):
    x = tf.placeholder(
      self.params['dtype'],
      [self.params['batch_size'], None,
       self.params['num_audio_features']],
      name="ph_x",
    )
    x_length = tf.placeholder(
      tf.int32,
      [self.params['batch_size']],
      name="ph_xlen",
    )
    if self.params['use_targets']:
      y = tf.placeholder(
        tf.int32,
        [self.params['batch_size'], None],
        name="ph_y",
      )
      y_length = tf.placeholder(
        tf.int32,
        [self.params['batch_size']],
        name="ph_ylen",
      )
      return [x, x_length, y, y_length]
    else:
      return [x, x_length]

  def get_size_in_samples(self):
    return 10000

  def next_batch_feed_dict(self):
    if self.random_data is None:
      self.random_data = [None] * 4
      seq_length = 2048
      self.random_data[0] = np.random.rand(
        self.params['batch_size'], seq_length, self.params['num_audio_features']
      )
      self.random_data[1] = np.ones(self.params['batch_size']) * seq_length
      tgt_length = 50
      self.random_data[2] = np.random.randint(
        0, 10, size=(self.params['batch_size'], tgt_length)
      )
      self.random_data[3] = np.ones(self.params['batch_size']) * tgt_length

    return {self.get_input_tensors(): self.random_data}


class Speech2TextTFDataLayer(DataLayer):
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'num_audio_features': int,
      'input_type': ['spectrogram', 'mfcc'],
      'alphabet_config_path': str,
      'dataset_path': list,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'augmentation': dict,
    })

  def __init__(self, params, model):
    """
    Required params:
      batch_size
      alphabet_config_path
      dataset_path
      num_audio_features
      input_type
    Example:
      params = {
        'batch_size': 16,
        'alphabet_config_path': 'data/alphabet.txt',
        'dataset_path': ['/raid/data/speech/WSJ/wsj-train-128.csv'],
        'num_audio_features': 161,
        'input_type': 'spectrogram',
      }
    """
    super(Speech2TextTFDataLayer, self).__init__(params, model)

    self.params['alphabet'] = Alphabet(
      os.path.abspath(params['alphabet_config_path'])
    )
    self.params['tgt_vocab_size'] = self.params['alphabet'].size() + 1

    self._files = None
    for csv in params['dataset_path']:
      files = pd.read_csv(csv, encoding='utf-8')
      if self._files is None:
        self._files = files
      else:
        self._files = self._files.append(files)

    if self.params['use_targets']:
      cols = ['wav_filename', 'transcript']
    else:
      cols = 'wav_filename'
    self._files = self._files.loc[:, cols].values
    self.params['files'] = self._files

    self._size = self.get_size_in_samples()

    self.tfdataset = tf.data.Dataset.from_tensor_slices(self._files)
    if self.params['shuffle'] and self.params['use_targets']:
      self.tfdataset = self.tfdataset.shuffle(self._size)
    self.tfdataset = self.tfdataset.repeat()

    if self.params['use_targets']:
      self.tfdataset = self.tfdataset.map(lambda line:
        tf.py_func(self._parse_audio_transcript_element,
          [line], [self.params['dtype'], tf.int32, tf.int32, tf.int32],
          stateful=False
        ),
        num_parallel_calls=8
      )
      self.tfdataset = self.tfdataset.padded_batch(
        self.params['batch_size'],
        padded_shapes=([None, self.params['num_audio_features']], 1, [None], 1)
      )
    else:
      self.tfdataset = self.tfdataset.map(lambda line:
        tf.py_func(self._parse_audio_element,
          [line], [self.params['dtype'], tf.int32],
          stateful=False
        ),
        num_parallel_calls=8
      )
      self.tfdataset = self.tfdataset.padded_batch(
        self.params['batch_size'],
        padded_shapes=([None, self.params['num_audio_features']], 1)
      )

    self.iterator = self.tfdataset.prefetch(8).make_one_shot_iterator()

  def _parse_audio_transcript_element(self, element):
    """
    Parses tf.data element from TextLineDataset,
    returns source audio and target text NumPy arrays
    """
    audio_filename, transcript = element
    if not six.PY2:
      transcript = str(transcript, 'utf-8')
    target = text_to_char_array(transcript, self.params['alphabet'])
    source = get_speech_features_from_file(
      audio_filename, self.params['num_audio_features'],
      features_type=self.params['input_type'],
      augmentation=self.params.get('augmentation', None),
    )
    return source.astype(self.params['dtype'].as_numpy_dtype()), \
           np.int32([len(source)]), \
           np.int32(target), \
           np.int32([len(target)])

  def _parse_audio_element(self, audio_filename):
    """
    Parses tf.data element from TextLineDataset,
    returns source audio NumPy array
    """
    source = get_speech_features_from_file(
      audio_filename, self.params['num_audio_features'],
      features_type=self.params['input_type'],
      augmentation=self.params.get('augmentation', None),
    )
    return source.astype(self.params['dtype'].as_numpy_dtype()), \
           np.int32([len(source)])

  def gen_input_tensors(self):
    if self.params['use_targets']:
      x, x_length, y, y_length = self.iterator.get_next()
      # need to explicitly set batch size dimension (it is employed in the model)
      y.set_shape([self.params['batch_size'], None])
      y_length = tf.reshape(y_length, [self.params['batch_size']])
    else:
      x, x_length = self.iterator.get_next()
    x.set_shape([self.params['batch_size'], None,
                 self.params['num_audio_features']])
    x_length = tf.reshape(x_length, [self.params['batch_size']])

    if self.params['use_targets']:
      return [x, x_length, y, y_length]
    else:
      return [x, x_length]

  def shuffle(self):
    pass

  def get_size_in_samples(self):
    return len(self._files)

  def next_batch_feed_dict(self):
    return {}
