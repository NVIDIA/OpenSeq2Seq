# Copyright (c) 2018 NVIDIA Corporation
"""Data Layer for Speech-to-Text models"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
import six
from six import string_types
from six.moves import range
import inspect
import python_speech_features as psf

from open_seq2seq.data.data_layer import DataLayer
from open_seq2seq.data.utils import load_pre_existing_vocabulary
from .speech_utils import get_speech_features_from_file, get_speech_features
import sentencepiece as spm

# numpy.fft MKL bug: https://github.com/IntelPython/mkl_fft/issues/11
if hasattr(np.fft, 'restore_all'):
  np.fft.restore_all()

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
        'max_duration': float,
        'bpe': bool,
        'autoregressive': bool,
        'syn_enable': bool,
        'syn_subdirs': list,
        'window_size': float,
        'window_stride': float,
    })

  def __init__(self, params, model, num_workers, worker_id):
    """Speech-to-text data layer constructor.
    See parent class for arguments description.
    Config parameters:
    * **num_audio_features** (int) --- number of audio features to extract.
    * **input_type** (str) --- could be either "spectrogram" or "mfcc".
    * **vocab_file** (str) --- path to vocabulary file or sentencepiece model.
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
    * **autoregressive** (bool) --- boolean indicating whether the model is
      autoregressive.
    * **syn_enable** (bool) --- boolean indicating whether the model is
      using synthetic data.
    * **syn_subdirs** (list) --- must be defined if using synthetic mode.
      Contains a list of subdirectories that hold the synthetica wav files.
    """
    super(Speech2TextDataLayer, self).__init__(params, model,
                                               num_workers, worker_id)
    # we need this until python_speech_features gets update on pypi.org
    self.apply_window = 'winfunc' in inspect.getargspec(psf.logfbank)[0]
    if not self.apply_window and \
        (self.params['input_type'] == 'mfcc' or \
         self.params['input_type'] == 'logfbank'):
      print('WARNING: using python_speech_features WITHOUT windowing function')
      print('Please install the latest python_speech_features (from GitHub)')
    self.params['autoregressive'] = self.params.get('autoregressive', False)
    self.autoregressive = self.params['autoregressive']
    self.params['bpe'] = self.params.get('bpe', False)
    if self.params['bpe']:
      self.sp = spm.SentencePieceProcessor()
      self.sp.Load(self.params['vocab_file'])
      self.params['tgt_vocab_size'] = len(self.sp) + 1
    else:
      self.params['char2idx'] = load_pre_existing_vocabulary(
          self.params['vocab_file'], read_chars=True,
      )
      if not self.autoregressive:
        # add one for implied blank token
        self.params['tgt_vocab_size'] = len(self.params['char2idx']) + 1
      else:
        num_chars_orig = len(self.params['char2idx'])
        self.params['tgt_vocab_size'] = num_chars_orig + 2
        self.start_index = num_chars_orig
        self.end_index = num_chars_orig + 1
        self.params['char2idx']['<S>'] = self.start_index
        self.params['char2idx']['</S>'] = self.end_index
        self.target_pad_value = self.end_index
      self.params['idx2char'] = {i: w for w,
                                 i in self.params['char2idx'].items()}
    self.target_pad_value = 0

    self._files = None
    if self.params["interactive"]:
      return
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

    self.params['max_duration'] = params.get('max_duration', -1.0)
    self.params['window_size'] = params.get('window_size', 20e-3)
    self.params['window_stride'] = params.get('window_stride', 10e-3)

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
  def iterator(self):
    """Underlying tf.data iterator."""
    return self._iterator

  def build_graph(self):
    with tf.device('/cpu:0'):

      """Builds data processing graph using ``tf.data`` API."""
      if self.params['mode'] != 'infer':
        self._dataset = tf.data.Dataset.from_tensor_slices(self._files)
        if self.params['shuffle']:
          self._dataset = self._dataset.shuffle(self._size)
        self._dataset = self._dataset.repeat()
        self._dataset = self._dataset.prefetch(tf.contrib.data.AUTOTUNE)
        self._dataset = self._dataset.map(
            lambda line: tf.py_func(
                self._parse_audio_transcript_element,
                [line],
                [self.params['dtype'], tf.int32, tf.int32, tf.int32, tf.float32],
                stateful=False,
            ),
            num_parallel_calls=8,
        )
        if self.params['max_duration'] > 0:
          self._dataset = self._dataset.filter(
              lambda x, x_len, y, y_len, duration:
              tf.less_equal(duration, self.params['max_duration'])
          )
        self._dataset = self._dataset.map(
            lambda x, x_len, y, y_len, duration:
            [x, x_len, y, y_len],
            num_parallel_calls=8,
        )
        self._dataset = self._dataset.padded_batch(
            self.params['batch_size'],
            padded_shapes=([None, self.params['num_audio_features']],
                           1, [None], 1),
            padding_values=(
                tf.cast(0, self.params['dtype']), 0, self.target_pad_value, 0),
        )
      else:
        indices = self.split_data(
            np.array(list(map(str, range(len(self.all_files)))))
        )
        self._dataset = tf.data.Dataset.from_tensor_slices(
            np.hstack((indices[:, np.newaxis], self._files[:, np.newaxis]))
        )
        self._dataset = self._dataset.repeat()
        self._dataset = self._dataset.prefetch(tf.contrib.data.AUTOTUNE)
        self._dataset = self._dataset.map(
            lambda line: tf.py_func(
                self._parse_audio_element,
                [line],
                [self.params['dtype'], tf.int32, tf.int32, tf.float32],
                stateful=False,
            ),
            num_parallel_calls=8,
        )
        if self.params['max_duration'] > 0:
          self._dataset = self._dataset.filter(
              lambda x, x_len, idx, duration:
              tf.less_equal(duration, self.params['max_duration'])
          )
        self._dataset = self._dataset.map(
            lambda x, x_len, idx, duration:
            [x, x_len, idx],
            num_parallel_calls=8,
        )
        self._dataset = self._dataset.padded_batch(
            self.params['batch_size'],
            padded_shapes=([None, self.params['num_audio_features']], 1, 1)
        )

      self._iterator = self._dataset.prefetch(tf.contrib.data.AUTOTUNE)\
                           .make_initializable_iterator()

      if self.params['mode'] != 'infer':
        x, x_length, y, y_length = self._iterator.get_next()
        # need to explicitly set batch size dimension
        # (it is employed in the model)
        y.set_shape([self.params['batch_size'], None])
        y_length = tf.reshape(y_length, [self.params['batch_size']])
      else:
        x, x_length, x_id = self._iterator.get_next()
        x_id = tf.reshape(x_id, [self.params['batch_size']])

      x.set_shape([self.params['batch_size'], None,
                   self.params['num_audio_features']])
      x_length = tf.reshape(x_length, [self.params['batch_size']])

      self._input_tensors = {}
      self._input_tensors["source_tensors"] = [x, x_length]
      if self.params['mode'] != 'infer':
        self._input_tensors['target_tensors'] = [y, y_length]
      else:
        self._input_tensors['source_ids'] = [x_id]

  def create_interactive_placeholders(self):
    self._x = tf.placeholder(
        dtype=self.params['dtype'],
        shape=[
            self.params['batch_size'],
            None,
            self.params['num_audio_features']
        ]
    )
    self._x_length = tf.placeholder(
        dtype=tf.int32,
        shape=[self.params['batch_size']]
    )
    self._x_id = tf.placeholder(
        dtype=tf.int32,
        shape=[self.params['batch_size']]
    )

    self._input_tensors = {}
    self._input_tensors["source_tensors"] = [self._x, self._x_length]
    self._input_tensors['source_ids'] = [self._x_id]

  def create_feed_dict(self, model_in):
    """ Creates the feed dict for interactive infer

    Args:
      model_in (str or np.array): Either a str that contains the file path of the
        wav file, or a numpy array containing 1-d wav file.

    Returns:
      feed_dict (dict): Dictionary with values for the placeholders.
    """
    audio_arr = []
    audio_length_arr = []
    x_id_arr = []
    for line in model_in:
      if isinstance(line, string_types):
        audio, audio_length, x_id, _ = self._parse_audio_element([0, line])
      elif isinstance(line, np.ndarray):
        audio, audio_length, x_id, _ = self._get_audio(line)
      else:
        raise ValueError(
            "Speech2Text's interactive inference mode only supports string or",
            "numpy array as input. Got {}". format(type(line))
        )
      audio_arr.append(audio)
      audio_length_arr.append(audio_length)
      x_id_arr.append(x_id)
    max_len = np.max(audio_length_arr)
    for i, audio in enumerate(audio_arr):
      audio = np.pad(
          audio, ((0, max_len-len(audio)), (0,0)),
          "constant", constant_values=0.
      )
      audio_arr[i] = audio

    audio = np.reshape(
        audio_arr,
        [self.params['batch_size'],
         -1,
         self.params['num_audio_features']]
    )
    audio_length = np.reshape(audio_length_arr, [self.params['batch_size']])
    x_id = np.reshape(x_id_arr, [self.params['batch_size']])

    feed_dict = {
        self._x: audio,
        self._x_length: audio_length,
        self._x_id: x_id,
    }
    return feed_dict

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
      audio_filename = str(audio_filename, 'utf-8')
    if self.params['bpe']:
      target_indices = self.sp.EncodeAsIds(transcript)
    else:
      target_indices = [self.params['char2idx'][c] for c in transcript]
    if self.autoregressive:
      target_indices = target_indices + [self.end_index]
    target = np.array(target_indices)

    if self.params.get("syn_enable", False):
      audio_filename = audio_filename.format(np.random.choice(self.params["syn_subdirs"]))

    source, audio_duration = get_speech_features_from_file(
        audio_filename, self.params['num_audio_features'],
        pad_to=self.params.get('pad_to', 8),
        features_type=self.params['input_type'],
        window_size=self.params['window_size'],
        window_stride=self.params['window_stride'],
        augmentation=self.params.get('augmentation', None),
        apply_window=self.apply_window,
        cache_features=self.params.get('cache_features', False),
        cache_format=self.params.get('cache_format', 'hdf5'),
        cache_regenerate=self.params.get('cache_regenerate', False),
        params=self.params
    )
    return source.astype(self.params['dtype'].as_numpy_dtype()), \
        np.int32([len(source)]), \
        np.int32(target), \
        np.int32([len(target)]), \
        np.float32([audio_duration])

  def _get_audio(self, wav):
    """Parses audio from wav and returns array of audio features.
    Args:
      wav: numpy array containing wav

    Returns:
      tuple: source audio features as ``np.array``, length of source sequence,
      sample id.
    """
    pad_to = self.params.get('pad_to', 8)
    source, audio_duration = get_speech_features(
        wav, 16000., self.params['num_audio_features'], pad_to,
        features_type=self.params['input_type'],
        window_size=self.params['window_size'],
        window_stride=self.params['window_stride'],
        augmentation=self.params.get('augmentation', None),
        apply_window=self.apply_window
    )
    return source.astype(self.params['dtype'].as_numpy_dtype()), \
        np.int32([len(source)]), np.int32([0]), \
        np.float32([audio_duration])

  def _parse_audio_element(self, id_and_audio_filename):
    """Parses audio from file and returns array of audio features.
    Args:
      id_and_audio_filename: tuple of sample id and corresponding
          audio file name.
    Returns:
      tuple: source audio features as ``np.array``, length of source sequence,
      sample id.
    """
    idx, audio_filename = id_and_audio_filename
    pad_to = self.params.get('pad_to', 8)
    source, audio_duration = get_speech_features_from_file(
        audio_filename, self.params['num_audio_features'], pad_to,
        features_type=self.params['input_type'],
        window_size=self.params['window_size'],
        window_stride=self.params['window_stride'],
        augmentation=self.params.get('augmentation', None),
        apply_window=self.apply_window
    )
    return source.astype(self.params['dtype'].as_numpy_dtype()), \
        np.int32([len(source)]), np.int32([idx]), \
        np.float32([audio_duration])

  @property
  def input_tensors(self):
    """Dictionary with input tensors.
    ``input_tensors["source_tensors"]`` contains:
      * source_sequence
        (shape=[batch_size x sequence length x num_audio_features])
      * source_length (shape=[batch_size])
    ``input_tensors["target_tensors"]`` contains:
      * target_sequence
        (shape=[batch_size x sequence length])
      * target_length (shape=[batch_size])
    """
    return self._input_tensors

  def get_size_in_samples(self):
    """Returns the number of audio files."""
    return len(self._files)

