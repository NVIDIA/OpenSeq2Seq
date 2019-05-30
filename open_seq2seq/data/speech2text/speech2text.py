# Copyright (c) 2018 NVIDIA Corporation
"""Data Layer for Speech-to-Text models"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
import six
import math
import librosa
from six import string_types
from six.moves import range

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
        'backend': ['psf', 'librosa'],
        'augmentation': dict,
        'pad_to': int,
        'max_duration': float,
        'min_duration': float,
        'bpe': bool,
        'autoregressive': bool,
        'syn_enable': bool,
        'syn_subdirs': list,
        'window_size': float,
        'window_stride': float,
        'dither': float,
        'norm_per_feature': bool,
        'window': ['hanning', 'hamming', 'none'],
        'num_fft': int,
        'precompute_mel_basis': bool,
        'sample_freq': int,
    })

  def __init__(self, params, model, num_workers, worker_id):
    """Speech-to-text data layer constructor.
    See parent class for arguments description.
    Config parameters:
    * **backend** (str) --- audio pre-processing backend
      ('psf' [default] or librosa [recommended]).
    * **num_audio_features** (int) --- number of audio features to extract.
    * **input_type** (str) --- could be either "spectrogram" or "mfcc".
    * **vocab_file** (str) --- path to vocabulary file or sentencepiece model.
    * **dataset_files** (list) --- list with paths to all dataset .csv files.
    * **augmentation** (dict) --- optional dictionary with data augmentation
      parameters. Can contain "speed_perturbation_ratio", "noise_level_min" and
      "noise_level_max" parameters, e.g.::
        {
          'speed_perturbation_ratio': 0.05,
          'noise_level_min': -90,
          'noise_level_max': -60,
        }
      For additional details on these parameters see
      :func:`data.speech2text.speech_utils.augment_audio_signal` function.
    * **pad_to** (int) --- align audio sequence length to pad_to value.
    * **max_duration** (float) --- drop all samples longer than
      **max_duration** (seconds)
    * **min_duration** (float) --- drop all samples shorter than
      **min_duration** (seconds)
    * **bpe** (bool) --- use BPE encodings
    * **autoregressive** (bool) --- boolean indicating whether the model is
      autoregressive.
    * **syn_enable** (bool) --- boolean indicating whether the model is
      using synthetic data.
    * **syn_subdirs** (list) --- must be defined if using synthetic mode.
      Contains a list of subdirectories that hold the synthetica wav files.
    * **window_size** (float) --- window's duration (in seconds)
    * **window_stride** (float) --- window's stride (in seconds)
    * **dither** (float) --- weight of Gaussian noise to apply to input signal
      for dithering/preventing quantization noise
    * **num_fft** (int) --- size of fft window to use if features require fft,
          defaults to smallest power of 2 larger than window size
    * **norm_per_feature** (bool) --- if True, the output features will be
      normalized (whitened) individually. if False, a global mean/std over all
      features will be used for normalization.
    * **window** (str) --- window function to apply before FFT
      ('hanning', 'hamming', 'none')
    * **num_fft** (int) --- optional FFT size
    * **precompute_mel_basis** (bool) --- compute and store mel basis. If False,
      it will compute it for every get_speech_features call. Default: False
    * **sample_freq** (int) --- required for precompute_mel_basis
    """
    super(Speech2TextDataLayer, self).__init__(params, model,
                                               num_workers, worker_id)
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

    self.params['min_duration'] = self.params.get('min_duration', -1.0)
    self.params['max_duration'] = self.params.get('max_duration', -1.0)
    self.params['window_size'] = self.params.get('window_size', 20e-3)
    self.params['window_stride'] = self.params.get('window_stride', 10e-3)

    mel_basis = None
    if (self.params.get("precompute_mel_basis", False) and
        self.params["input_type"] == "logfbank"):
      num_fft = (
          self.params.get("num_fft", None) or
          2**math.ceil(math.log2(
              self.params['window_size']*self.params["sample_freq"])
          )
      )
      mel_basis = librosa.filters.mel(
          self.params["sample_freq"],
          num_fft,
          n_mels=self.params["num_audio_features"],
          fmin=0,
          fmax=int(self.params["sample_freq"]/2)
      )
    self.params['mel_basis'] = mel_basis

    if 'n_freq_mask' in self.params.get('augmentation', {}):
      width_freq_mask = self.params['augmentation'].get('width_freq_mask', 10)
      if width_freq_mask > self.params['num_audio_features']:
        raise ValueError(
            "'width_freq_mask'={} should be smaller ".format(width_freq_mask)+
            "than 'num_audio_features'={}".format(
               self.params['num_audio_features']
            )
        )


    if 'time_stretch_ratio' in self.params.get('augmentation', {}):
      print("WARNING: Please update time_stretch_ratio to speed_perturbation_ratio")
      self.params['augmentation']['speed_perturbation_ratio'] = self.params['augmentation']['time_stretch_ratio']

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
        if self.params['min_duration'] > 0:
          self._dataset = self._dataset.filter(
              lambda x, x_len, y, y_len, duration:
              tf.greater_equal(duration, self.params['min_duration'])
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
        if self.params['min_duration'] > 0:
            self._dataset = self._dataset.filter(
              lambda x, x_len, y, y_len, duration:
              tf.greater_equal(duration, self.params['min_duration'])
          )
        self._dataset = self._dataset.map(
            lambda x, x_len, idx, duration:
            [x, x_len, idx],
            num_parallel_calls=16,
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

      pad_to = self.params.get("pad_to", 8)
      if pad_to > 0 and self.params.get('backend') == 'librosa':
        # we do padding with TF for librosa backend
        num_pad = tf.mod(pad_to - tf.mod(tf.reduce_max(x_length), pad_to), pad_to)
        x = tf.pad(x, [[0, 0], [0, num_pad], [0, 0]])

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
    pad_to = self.params.get("pad_to", 8)
    if pad_to > 0 and self.params.get('backend') == 'librosa':
      max_len += (pad_to - max_len % pad_to) % pad_to

    for i, audio in enumerate(audio_arr):
      audio = np.pad(
          audio, ((0, max_len-len(audio)), (0, 0)),
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
        audio_filename,
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
    source, audio_duration = get_speech_features(
        wav, 16000., self.params
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
    source, audio_duration = get_speech_features_from_file(
        audio_filename,
        params=self.params
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
