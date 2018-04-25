# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import six
import numpy as np
import tensorflow as tf
import pandas as pd

from .data_layer import DataLayer
from .speech_utils import get_speech_features_from_file
from .utils import load_pre_existing_vocabulary


class Speech2TextPlaceholdersDataLayer(DataLayer):
  """Speech-to-text data layer class, that **does not** use ``tf.data`` API.
  This class should not be used in real experiments, since it is a lot slower
  than fast ``tf.data`` based implementation. It can be useful in debugging
  certain things and as an example of data layer with placeholders.
  """
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'num_audio_features': int,
      'input_type': ['spectrogram', 'mfcc'],
      'vocab_file': str,
      'dataset_files': list,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'augmentation': dict,
    })

  def __init__(self, params, model, num_workers=None, worker_id=None):
    """Speech-to-text placeholders-based data layer constructor.

    See parent class for argument description.

    Config parameters:

    * **num_audio_features** (int) --- number of audio features to extract.
    * **input_type** (str) --- could be either "spectrogram" or "mfcc".
    * **vocab_file** (str) --- path to vocabulary file.
    * **dataset_files** (list) --- list with paths to all dataset .csv files.
    * **augmentation** (dict) --- optional dictionary with data augmentation
      parameters. Can contain "time_stretch_ratio", "noise_level_min" and
      "noise_level_max" parameters, e.g.::
        {
          'time_stretch_ratio': 0.05,
          'noise_level_min': -90,
          'noise_level_max': -60,
        }
      For additional details on these parameters see
      :func:`data.speech_utils.augment_audio_signal` function.
    """
    super(Speech2TextPlaceholdersDataLayer, self).__init__(params, model,
                                                           num_workers, worker_id)

    self.params['char2idx'] = load_pre_existing_vocabulary(
      self.params['vocab_file'], read_chars=True,
    )
    self.params['idx2char'] = {i: w for w, i in self.params['char2idx'].items()}
    # add one for implied blank token
    self.params['tgt_vocab_size'] = len(self.params['char2idx']) + 1

    self._index = -1
    self._files = None
    for csv in params['dataset_files']:
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

  def build_graph(self):
    """Empty since no graph construction is required,
    besides creating placeholders.
    """
    pass

  def shuffle(self):
    """Shuffles list of file names."""
    self._files = np.random.permutation(self._files)

  def gen_input_tensors(self):
    """Creates and returns necessary placeholders."""
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
    """Returns the number of audio files."""
    return len(self._files)

  def get_one_sample(self):
    """This is a helper function that processes one audio file
    and its transcript.
    """
    self._index += 1
    if self._index == self._size:
      self._index = 0

    if self.params['use_targets']:
      wav_file, transcript = self._files[self._index]
      target = np.array([self.params['char2idx'][c] for c in transcript])
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
    """Returns next batch data in numpy arrays."""
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
    """Generates next batch feed dictionary."""
    return {self.get_input_tensors(): self.next_batch()}


class Speech2TextRandomDataLayer(DataLayer):
  """This class should be used for performance profiling only.
  It generates random sequences instead of the real audio data.
  """
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'num_audio_features': int,
      'input_type': ['spectrogram', 'mfcc'],
      'vocab_file': str,
      'dataset_files': list,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'augmentation': dict,
    })

  def __init__(self, params, model, num_workers=None, worker_id=None):
    super(Speech2TextRandomDataLayer, self).__init__(params, model,
                                                     num_workers, worker_id)
    self.random_data = None
    self.params['char2idx'] = load_pre_existing_vocabulary(
      self.params['vocab_file'], read_chars=True,
    )
    self.params['idx2char'] = {i: w for w, i in self.params['char2idx'].items()}
    # add one for implied blank token
    self.params['tgt_vocab_size'] = len(self.params['char2idx']) + 1

  def build_graph(self):
    pass

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


class Speech2TextDataLayer(DataLayer):
  """Speech-to-text data layer class, that uses ``tf.data`` API.
  This is a recommended class to use in real experiments.
  """
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'num_audio_features': int,
      'input_type': ['spectrogram', 'mfcc'],
      'vocab_file': str,
      'dataset_files': list,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'augmentation': dict,
    })

  def __init__(self, params, model, num_workers=None, worker_id=None):
    """Speech-to-text ``tf.data`` based data layer constructor.

    See parent class for argument description.

    Config parameters:

    * **num_audio_features** (int) --- number of audio features to extract.
    * **input_type** (str) --- could be either "spectrogram" or "mfcc".
    * **vocab_file** (str) --- path to vocabulary file.
    * **dataset_files** (list) --- list with paths to all dataset .csv files.
    * **augmentation** (dict) --- optional dictionary with data augmentation
      parameters. Can contain "time_stretch_ratio", "noise_level_min" and
      "noise_level_max" parameters, e.g.::
        {
          'time_stretch_ratio': 0.05,
          'noise_level_min': -90,
          'noise_level_max': -60,
        }
      For additional details on these parameters see
      :func:`data.speech_utils.augment_audio_signal` function.
    """
    super(Speech2TextDataLayer, self).__init__(params, model,
                                               num_workers, worker_id)

    self.params['char2idx'] = load_pre_existing_vocabulary(
      self.params['vocab_file'], read_chars=True,
    )
    self.params['idx2char'] = {i: w for w, i in self.params['char2idx'].items()}
    # add one for implied blank token
    self.params['tgt_vocab_size'] = len(self.params['char2idx']) + 1

    self._files = None
    for csv in params['dataset_files']:
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
    self.tfdataset = None
    self.iterator = None

  def build_graph(self):
    """Builds data reading graph using ``tf.data`` API."""
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
    """Parses tf.data element from TextLineDataset into audio and text.

    Args:
      element: tf.data element from TextLineDataset.

    Returns:
      tuple: source audio features as ``np.array``, length of source sequence,
      target text as `np.array` of ids, target text length.
    """
    audio_filename, transcript = element
    if not six.PY2:
      transcript = str(transcript, 'utf-8')
    target = np.array([self.params['char2idx'][c] for c in transcript])
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
    """Parses audio from file and returns array of audio features.

    Args:
      audio_filename: audio file name.

    Returns:
      tuple: source audio features as ``np.array``, length of source sequence,
    """
    source = get_speech_features_from_file(
      audio_filename, self.params['num_audio_features'],
      features_type=self.params['input_type'],
      augmentation=self.params.get('augmentation', None),
    )
    return source.astype(self.params['dtype'].as_numpy_dtype()), \
           np.int32([len(source)])

  def gen_input_tensors(self):
    """Generates input tensors using ``iterator.get_next()`` method."""
    if self.params['use_targets']:
      x, x_length, y, y_length = self.iterator.get_next()
      # need to explicitly set batch size dimension
      # (it is employed in the model)
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
    """This method is empty, since shuffling is performed by ``tf.data``."""
    pass

  def get_size_in_samples(self):
    """Returns the number of audio files."""
    return len(self._files)

  def next_batch_feed_dict(self):
    """This method is empty, since ``tf.data`` does not need feed dictionary."""
    return {}
