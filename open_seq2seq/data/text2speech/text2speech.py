# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import string
import os
import six
import numpy as np
import tensorflow as tf
import pandas as pd

from open_seq2seq.data.data_layer import DataLayer
from .speech_utils import get_speech_features_from_file
from open_seq2seq.data.utils import load_pre_existing_vocabulary


class Text2SpeechDataLayer(DataLayer):
  """Text-to-speech data layer class, that uses ``tf.data`` API.
  This is a recommended class to use in real experiments.
  """
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'num_audio_features': int,
      # 'input_type': ['spectrogram', 'mfcc'],
      'output_type': ['spectrogram', 'mfcc', 'mel', 'test', 'spectrogram_disk'],
      'vocab_file': str,
      'dataset_files': list,
      'dataset_location': str,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'augmentation': dict,
      'pad_to': int,
      'mag_power': int,
      'feature_normalize': bool,
      'pad_EOS': bool
    })

  def __init__(self, params, model, num_workers=None, worker_id=None):
    """Text-to-speech ``tf.data`` based data layer constructor.

    See parent class for arguments description.

    Config parameters:

    * **num_audio_features** (int) --- number of audio features to extract.
    * **output_type** (str) --- could be either "spectrogram" or "mfcc".
    * **vocab_file** (str) --- path to vocabulary file.
    * **dataset_files** (list) --- list with paths to all dataset .csv files.
    * **dataset_location** (string) --- string with path to directory where wavs are stored.
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
    super(Text2SpeechDataLayer, self).__init__(params, model,
                                               num_workers, worker_id)

    # There is no vocab file atm
    # Character level vocab
    self.params['char2idx'] = load_pre_existing_vocabulary(
      self.params['vocab_file'], read_chars=True,
    )
    self.params['idx2char'] = {i: w for w, i in self.params['char2idx'].items()}
    # add one for implied blank token
    self.params['src_vocab_size'] = len(self.params['char2idx']) + 1
    self.params['tgt_vocab_size'] = len(self.params['char2idx']) + 1

    names = ['wav_filename', 'transcript', 'transcript_normalized']

    if "disk" in self.params["output_type"]:
      self.load_from_disk = True
    else:
      self.load_from_disk = False

    self._files = None
    for csvs in params['dataset_files']:
      files = pd.read_csv(csvs, encoding='utf-8', sep='\x7c', header=None, names=names, quoting=3)
      if self._files is None:
        self._files = files
      else:
        self._files = self._files.append(files)

    if self.params['mode'] != 'infer':
      cols = ['wav_filename', 'transcript_normalized']
    else:
      cols = 'transcript_normalized'

    self.all_files = self._files.loc[:, cols].values
    self._files = self.split_data(self.all_files)

    self._size = self.get_size_in_samples()
    self._dataset = None
    self._iterator = None
    self._input_tensors = None

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
    return self._iterator

  def build_graph(self):
    """Builds data reading graph using ``tf.data`` API."""
    self._dataset = tf.data.Dataset.from_tensor_slices(self._files)
    if self.params['shuffle']:
      self._dataset = self._dataset.shuffle(self._size)
    self._dataset = self._dataset.repeat()

    if self.params['mode'] != 'infer':
      self._dataset = self._dataset.map(
        lambda line: tf.py_func(
          self._parse_audio_transcript_element,
          [line],
          [tf.int32, tf.int32, self.params['dtype'], tf.int32],
          stateful=False,
        ),
        num_parallel_calls=8,
      )
      self._dataset = self._dataset.padded_batch(
        self.params['batch_size'],
        padded_shapes=([None], 1, [None, self.params['num_audio_features']], 1)
      )
    else:
      self._dataset = self._dataset.map(
        lambda line: tf.py_func(
          self._parse_transcript_element,
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
      # y, y_length, x, x_length = self._iterator.get_next()
      # print(x.shape)
      # print(y.shape)
      # need to explicitly set batch size dimension
      # (it is employed in the model)
      y.set_shape([self.params['batch_size'], None,
                 self.params['num_audio_features']])
      y_length = tf.reshape(y_length, [self.params['batch_size']])
    else:
      x, x_length = self._iterator.get_next()
    x.set_shape([self.params['batch_size'], None])
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
      text_input text as `np.array` of ids, text_input text length.
    """
    audio_filename, transcript = element
    if not six.PY2:
      transcript = str(transcript, 'utf-8')
    # transcript = self._normalize_transcript(transcript)
    text_input = np.array([self.params['char2idx'][c] for c in unicode(transcript,"utf-8")])
    if self.params.get("pad_EOS", False):
      text_input.append(self.params['char2idx']["~"])
    pad_to = self.params.get('pad_to', 8)
    if self.load_from_disk:
      file_path = os.path.join(self.params['dataset_location'],audio_filename+".npy")
      spectrogram = np.load(file_path)
    else:
      file_path = os.path.join(self.params['dataset_location'],audio_filename+".wav")
      spectrogram = get_speech_features_from_file(
      file_path, self.params['num_audio_features'], pad_to,
      features_type=self.params['output_type'],
      augmentation=self.params.get('augmentation', None),
      mag_power=self.params.get('mag_power', 2),
      feature_normalize=self.params.get('feature_normalize', True),
      )
    if self.params.get("pad_EOS", False):
      spectrogram = np.pad(spectrogram, ((0,1),(0,0)), "constant", constant_values=0)
    return np.int32(text_input), \
           np.int32([len(text_input)]), \
           spectrogram.astype(self.params['dtype'].as_numpy_dtype()), \
           np.int32([len(spectrogram)])

  # Might not be useful for actual text2speech applications
  def _normalize_transcript(self, text):
    """Parses the transcript to remove punctation, lowercase all characters, and all non-ascii characters

    Args:
      text: the string to parse

    Returns:
      text: the normalized text
    """
    text = text.decode('utf-8').encode('ascii', errors="ignore")
    text = text.translate(None, string.punctuation)
    text = text.lower()
    return text

  def _parse_transcript_element(self, text):
    """Parses text from file and returns array of text features.

    Args:
      text: the string to parse.

    Returns:
      tuple: target text as `np.array` of ids, target text length.
    """
    if not six.PY2:
      transcript = str(transcript, 'utf-8')
    transcript = self._normalize_transcript(transcript)
    target = np.array([self.params['char2idx'][c] for c in transcript])

    return np.int32(target), \
           np.int32([len(target)])

  @property
  def input_tensors(self):
    return self._input_tensors

  def get_size_in_samples(self):
    """Returns the number of audio files."""
    return len(self._files)
