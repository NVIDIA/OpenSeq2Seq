# Copyright (c) 2018 NVIDIA Corporation
"""Data Layer for Speech-to-Text models"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io.wavfile as wave
import librosa
import six
from six import string_types
from six.moves import range

from python_speech_features.base import get_filterbanks
from open_seq2seq.data.data_layer import DataLayer
from open_seq2seq.data.utils import load_pre_existing_vocabulary
from .speech_utils import get_speech_features_from_file, get_speech_features, normalize_signal


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
        'autoregressive': bool,
    })

  def __init__(self, params, model, num_workers, worker_id):
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
    * **autoregressive** (bool) --- boolean indicating whether the model is autoregressive.  
    """
    super(Speech2TextDataLayer, self).__init__(params, model,
                                               num_workers, worker_id)

    self.params['autoregressive'] = self.params.get('autoregressive', False)
    self.autoregressive = self.params['autoregressive']
    self.params['char2idx'] = load_pre_existing_vocabulary(
        self.params['vocab_file'], read_chars=True,
    )
    self.target_pad_value = 0
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

    self.params['max_duration'] = params.get('max_duration', None)

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
      if self.params['max_duration'] is not None:
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
      ).cache()
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
      if self.params['max_duration'] is not None:
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
    target_indices = [self.params['char2idx'][c] for c in transcript]
    if self.autoregressive:
      target_indices = target_indices + [self.end_index]
    target = np.array(target_indices)

    if "gen_audio" in audio_filename:
      audio_filename = audio_filename.format(np.random.choice([46, 48, 50]))
    pad_to = self.params.get('pad_to', 8)
    source, audio_duration = get_speech_features_from_file(
        audio_filename, self.params['num_audio_features'], pad_to,
        features_type=self.params['input_type'],
        augmentation=self.params.get('augmentation', None),
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
        augmentation=self.params.get('augmentation', None),
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
        augmentation=self.params.get('augmentation', None),
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

class Speech2TextTensorFlowDataLayer(DataLayer):
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
        # 'augmentation': dict,
        # 'pad_to': int,
        'max_duration': float,
        'trim': bool,
    })

  def __init__(self, params, model, num_workers, worker_id):
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
    super(Speech2TextTensorFlowDataLayer, self).__init__(params, model,
                                               num_workers, worker_id)

    self.params['autoregressive'] = self.params.get('autoregressive', False)

    self.params['char2idx'] = load_pre_existing_vocabulary(
        self.params['vocab_file'], read_chars=True,
    )
    self.params['idx2char'] = {i: w for w,
                               i in self.params['char2idx'].items()}
    # add one for implied blank token
    self.params['tgt_vocab_size'] = len(self.params['char2idx']) + 1

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

    self.params['max_duration'] = params.get('max_duration', None)
    # self._mel_basis = get_filterbanks(64, 512, 16000, 0, 8000)

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
    """Builds data processing graph using ``tf.data`` API."""
    if self.params['mode'] != 'infer':
      self._dataset = tf.data.Dataset.from_tensor_slices(self._files)
      if self.params['shuffle']:
        self._dataset = self._dataset.shuffle(self._size)
      self._dataset = self._dataset.repeat()
      # self._dataset = self._dataset.prefetch(tf.contrib.data.AUTOTUNE)

      self._dataset = self._dataset.map(
          lambda line: tf.py_func(
              self._load_audio,
              [line],
              [tf.float32, tf.float32, tf.int32, tf.int32],
              # [self.params['dtype'], tf.int32, tf.int32, tf.int32, tf.float32],
              stateful=False,
          ),
          num_parallel_calls=8
      )
      if self.params['max_duration'] is not None:
        self._dataset = self._dataset.filter(
            lambda audio, audio_len, txt, txt_len:
            tf.less_equal(tf.div(audio_len, 16000.), self.params['max_duration'])
        )
      self._dataset = self._dataset.padded_batch(
          self.params['batch_size'],
          padded_shapes=([None], 1, [None], 1)
      )
      # self._dataset = self._dataset.map(self._get_spec_train).cache()
      # self._dataset = self._dataset.map(self._get_spec_train)
    else:
      indices = self.split_data(
          np.array(list(map(str, range(len(self.all_files)))))
      )
      self._dataset = tf.data.Dataset.from_tensor_slices(
          np.hstack((indices[:, np.newaxis], self._files[:, np.newaxis]))
      )
      self._dataset = self._dataset.repeat()
      self._dataset = self._dataset.map(
          lambda line: tf.py_func(
              self._parse_audio_element,
              [line],
              [tf.float32, tf.float32, tf.int32],
              # [tf.float32, tf.float32, tf.int32, tf.float32],
              stateful=False,
          ),
          num_parallel_calls=8,
      )
      if self.params['max_duration'] is not None:
        self._dataset = self._dataset.filter(
            # lambda audio, audio_len, idx, mel_basis:
            lambda audio, audio_len, idx:
            tf.less_equal(tf.div(audio_len, 16000.), self.params['max_duration'])
        )
      self._dataset = self._dataset.padded_batch(
          self.params['batch_size'],
          # padded_shapes=([None], 1, 1, [257,64])
          padded_shapes=([None], 1, 1)
      )
      # self._dataset = self._dataset.map(self._get_spec_infer)

    self._iterator = self._dataset.prefetch(tf.contrib.data.AUTOTUNE)\
                         .make_initializable_iterator()

    if self.params['mode'] != 'infer':
      x, x_length, y, y_length = self._iterator.get_next()
      # need to explicitly set batch size dimension
      # (it is employed in the model)
      y.set_shape([self.params['batch_size'], None])
      y_length = tf.reshape(y_length, [self.params['batch_size']])
    else:
      # x, x_length, x_id, mel_basis = self._iterator.get_next()
      x, x_length, x_id = self._iterator.get_next()
      x_id = tf.reshape(x_id, [self.params['batch_size']])

    # x, x_length = self._get_spec(x, x_length, mel_basis)
    x, x_length = self._get_spec(x, x_length)
    x.set_shape([self.params['batch_size'], None,
                 self.params['num_audio_features']])
    x_length = tf.reshape(x_length, [self.params['batch_size']])

    self._input_tensors = {}
    self._input_tensors["source_tensors"] = [x, x_length]
    if self.params['mode'] != 'infer':
      self._input_tensors['target_tensors'] = [y, y_length]
    else:
      self._input_tensors['source_ids'] = [x_id]

  def _load_audio(self, element):
    audio_filename, transcript = element
    if not six.PY2:
      transcript = str(transcript, 'utf-8')
      audio_filename = str(audio_filename, 'utf-8')
    target = np.array([self.params['char2idx'][c] for c in transcript])

    if "gen_audio" in audio_filename:
      audio_filename = audio_filename.format(np.random.choice([46, 48, 50]))
    if self.params.get("trim", False):
      signal, fs = librosa.core.load(audio_filename, sr=None)
      signal, _ = librosa.effects.trim(
          signal,
          frame_length=int(512/2),
          hop_length=int(512/4)
      )
    else:
      _, signal = wave.read(audio_filename)
    signal = (normalize_signal(signal.astype(np.float32)) * 32767.0).astype(np.float32)
    audio_len = len(signal)
    return np.float32(signal), \
        np.float32([audio_len]), \
        np.int32(target), \
        np.int32([len(target)])

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
    if self.params.get("trim", False):
      signal, fs = librosa.core.load(audio_filename, sr=None)
      signal, _ = librosa.effects.trim(
          signal,
          frame_length=int(512/2),
          hop_length=int(512/4)
      )
    else:
      _, signal = wave.read(audio_filename)
    signal = (normalize_signal(signal.astype(np.float32)) * 32767.0).astype(np.float32)
    audio_len = len(signal)
    return np.float32(signal), \
        np.float32([audio_len]), \
        np.int32([idx])
        # np.float32(self._mel_basis.T)

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
        augmentation=self.params.get('augmentation', None),
    )
    return source.astype(self.params['dtype'].as_numpy_dtype()), \
        np.int32([len(source)]), np.int32([0]), \
        np.float32([audio_duration])

  def _get_spec_train(self, wav_batch, wav_len_batch, txt, txt_len):
    wav_batch, wav_len_batch = self._get_spec(wav_batch, wav_len_batch)
    return wav_batch, wav_len_batch, txt, txt_len

  def _get_spec_infer(self, wav_batch, wav_len_batch, x_id):
    wav_batch, wav_len_batch = self._get_spec(wav_batch, wav_len_batch)
    return wav_batch, wav_len_batch, x_id

  # def _get_spec(self, wav_batch, wav_len_batch, mel_basis):
  def _get_spec(self, wav_batch, wav_len_batch):
    wav_batch.set_shape([self.params["batch_size"], None])
    # mel_basis.set_shape([self.params["batch_size"], 257, 64])
    complex_spec = tf.contrib.signal.stft(
        wav_batch, frame_length=320, frame_step=160, fft_length=512
    )
    log_spec_len = tf.cast(
        tf.floor(
            (wav_len_batch - 320. + 160.) / 160.
        ),
        tf.int32
    )
    log_offset = 1e-6

    # energy spec
    # spec = tf.abs(complex_spec)

    # power spec
    spec = tf.real(complex_spec * tf.conj(complex_spec))

    # If not mag, then either mel or mfcc
    # to get mfcc, we need log mel to begin with
    # if self.params["input_type"] == "logfbank" or self.params["input_type"] == "mfcc":
    if self.params["input_type"] is not "spectrogram":
      num_spec_bins = spec.shape[-1].value
      num_mel_bins = 64
      if self.params["input_type"] is "logfbank":
        num_mel_bins = self.params["num_audio_features"]
      low_freq, high_freq = 0., 8000.
      mel_basis = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spec_bins, 16000, low_freq,
        high_freq)
      mel_basis = tf.tile(tf.expand_dims(mel_basis, 0), [self.params["batch_size"], 1, 1])
      spec = tf.matmul(spec, mel_basis)

      spec.set_shape(spec.shape[:-1].concatenate(mel_basis.shape[-1:]))
      log_spec = tf.log(spec + log_offset)
      # If not mag and not mel, then get mfcc features
      if self.params["input_type"] == "mfcc":
        log_spec = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
          log_spec)[..., :self.params["num_audio_features"]]
    # Else if it is mag, get log mag
    else:
      log_spec = tf.log(spec + log_offset)

    # mean, variance = tf.nn.moments(log_spec, axes=[1, 2])
    # mean = tf.expand_dims(tf.expand_dims(mean, 1), 2)
    # variance = tf.expand_dims(tf.expand_dims(variance, 1), 2)
    # log_spec = (log_spec - mean) / tf.sqrt(variance)
    log_spec_len.set_shape([self.params["batch_size"], 1])
    log_spec = self._normalize_spec(log_spec, log_spec_len, [1, 2])

    return tf.cast(log_spec, self.params["dtype"]), \
      log_spec_len

  def _normalize_spec(self, specs, lengths, axes):
    def _safe_mean(var, count, axes):
      total_loss = tf.reduce_sum(var, axis=axes, keep_dims=True)
      return tf.div(total_loss, count)
    lengths = tf.squeeze(lengths, 1)
    mask = tf.sequence_mask(lengths=lengths, dtype=tf.float32)
    mask = tf.expand_dims(mask, -1)
    count = tf.reduce_sum(tf.multiply(tf.ones_like(specs), mask), axes, keep_dims=True)
    specs_zeroed = tf.multiply(specs, mask)
    mean = _safe_mean(specs_zeroed, count, axes)
    mean_diff = tf.squared_difference(specs, tf.stop_gradient(mean))
    mean_diff_zeroed = tf.multiply(mean_diff, mask)
    variance = _safe_mean(mean_diff_zeroed, count, axes)
    specs = (specs - mean) / tf.sqrt(variance)
    return specs

  @property
  def input_tensors(self):
    """Dictionary with input tensors.
    ``input_tensors["source_tensors"]`` contains:
      * source_sequence
        (shape=[batch_size x sequence length x num_audio_features])
      * source_length (shape=[batch_size])
    ``input_tensors["target_tensors"]`` contains:
      * target_sequence
        (shape=[batch_size x sequence length x num_audio_features])
      * target_length (shape=[batch_size])
    """
    return self._input_tensors

  def get_size_in_samples(self):
    """Returns the number of audio files."""
    return len(self._files)
