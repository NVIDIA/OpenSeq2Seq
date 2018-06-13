# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import six
import numpy as np
import tensorflow as tf
import pandas as pd

from open_seq2seq.data.data_layer import DataLayer
from .speech_utils import get_speech_features_from_file
from open_seq2seq.data.utils import load_pre_existing_vocabulary


class Speech2TextDataLayer(DataLayer):
  """Speech-to-text data layer class."""
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'num_audio_features': int,
      'input_type': ['spectrogram', 'mfcc', 'logfbank'],
      'vocab_file': str,
      'dataset_files': list,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'augmentation': dict,
      'pad_to': int,
    })

  def __init__(self, params, model, num_workers=None, worker_id=None):
    """Speech-to-text data layer constructor.

    See parent class for arguments description.

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
      :func:`data.speech2text.speech_utils.augment_audio_signal` function.
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

    if self.params['mode'] != 'infer':
      cols = ['wav_filename', 'transcript']
    else:
      cols = 'wav_filename'

    self.all_files = self._files.loc[:, cols].values
    self._files = self.split_data(self.all_files)

    self._size = self.get_size_in_samples()
    self._dataset = None
    self._iterator = None
    self._input_tensors = None

  def split_data(self, data):
    """Method that performs data split for evaluation."""
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
  def iterator(self):
    """Underlying tf.data iterator."""
    return self._iterator

  def build_graph(self):
    """Builds data processing graph using ``tf.data`` API."""
    self._dataset = tf.data.Dataset.from_tensor_slices(self._files)
    if self.params['shuffle']:
      self._dataset = self._dataset.shuffle(self._size)
    self._dataset = self._dataset.repeat()

    if self.params['mode'] != 'infer':
      self._dataset = self._dataset.map(
        lambda line: tf.py_func(
          self._parse_audio_transcript_element,
          [line],
          [self.params['dtype'], tf.int32, tf.int32, tf.int32],
          stateful=False,
        ),
        num_parallel_calls=8,
      )
      self._dataset = self._dataset.padded_batch(
        self.params['batch_size'],
        padded_shapes=([None, self.params['num_audio_features']], 1, [None], 1)
      )
    else:
      self._dataset = self._dataset.map(
        lambda line: tf.py_func(
          self._parse_audio_element,
          [line],
          [self.params['dtype'], tf.int32],
          stateful=False,
        ),
        num_parallel_calls=8,
      )
      self._dataset = self._dataset.padded_batch(
        self.params['batch_size'],
        padded_shapes=([None, self.params['num_audio_features']], 1)
      )

    self._iterator = self._dataset.prefetch(8).make_initializable_iterator()

    if self.params['mode'] != 'infer':
      x, x_length, y, y_length = self._iterator.get_next()
      # need to explicitly set batch size dimension
      # (it is employed in the model)
      y.set_shape([self.params['batch_size'], None])
      y_length = tf.reshape(y_length, [self.params['batch_size']])
    else:
      x, x_length = self._iterator.get_next()
    x.set_shape([self.params['batch_size'], None,
                 self.params['num_audio_features']])
    x_length = tf.reshape(x_length, [self.params['batch_size']])

    self._input_tensors = {}
    self._input_tensors["source_tensors"] = [x, x_length]
    if self.params['mode'] != 'infer':
      self._input_tensors['target_tensors'] = [y, y_length]

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
    pad_to = self.params.get('pad_to', 8)
    source = get_speech_features_from_file(
      audio_filename, self.params['num_audio_features'], pad_to,
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
    pad_to = self.params.get('pad_to', 8)
    source = get_speech_features_from_file(
      audio_filename, self.params['num_audio_features'], pad_to,
      features_type=self.params['input_type'],
      augmentation=self.params.get('augmentation', None),
    )
    return source.astype(self.params['dtype'].as_numpy_dtype()), \
           np.int32([len(source)])

  @property
  def input_tensors(self):
    """Dictionary with input tensors.

    ``input_tensors["source_tensors"]`` contains:

      * source_sequence (shape=[batch_size x sequence length x num_audio_features])
      * source_length (shape=[batch_size])

    ``input_tensors["target_tensors"]`` contains:

      * target_sequence (shape=[batch_size x sequence length x num_audio_features])
      * target_length (shape=[batch_size])
    """
    return self._input_tensors

  def get_size_in_samples(self):
    """Returns the number of audio files."""
    return len(self._files)
