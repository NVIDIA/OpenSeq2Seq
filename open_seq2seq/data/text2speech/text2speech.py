# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import string
import os
import six
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd

from open_seq2seq.data.data_layer import DataLayer
from .speech_utils import get_speech_features_from_file, get_mel, inverse_mel, normalize, denormalize
from open_seq2seq.data.utils import load_pre_existing_vocabulary


class Text2SpeechDataLayer(DataLayer):
  """Text-to-speech data layer class
  """
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'num_audio_features': int,
      'output_type': ['magnitude', 'mel', 'magnitude_disk', 'mel_disk'],
      'vocab_file': str,
      'dataset_files': list,
      'dataset_location': str,
      'feature_normalize': bool,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'pad_to': int,
      'mag_power': int,
      'pad_EOS': bool,
      'feature_normalize_mean': float,
      'feature_normalize_std': float
    })

  def __init__(self, params, model, num_workers=None, worker_id=None):
    """Text-to-speech data layer constructor.

    See parent class for arguments description.

    Config parameters:

    * **num_audio_features** (int) --- number of audio features to extract.
    * **output_type** (str) --- could be either "magnitude", or "mel".
    * **vocab_file** (str) --- path to vocabulary file.
    * **dataset_files** (list) --- list with paths to all dataset .csv files. File is assumed
      to be separated by "|".
    * **dataset_location** (string) --- string with path to directory where wavs are stored.
    * **mag_power** (int) --- the power to which the magnitude spectrogram is scaled to
      1 for energy spectrogram
      2 for power spectrogram
      Defaults to 2.
    * **pad_EOS** (bool) --- whether to apply EOS tokens to both the text and the speech signal.
      Defaults to True.
    """
    super(Text2SpeechDataLayer, self).__init__(params, model,
                                               num_workers, worker_id)
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

    if "mel" in self.params["output_type"]:
      self.mel = True
      self.mel_basis = librosa.filters.mel(22050, 1024,
       n_mels=self.params['num_audio_features'])
    else:
      self.mel = False

    # Load csv files
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
    """Builds data reading graph."""
    self._dataset = tf.data.Dataset.from_tensor_slices(self._files)
    if self.params['shuffle']:
      self._dataset = self._dataset.shuffle(self._size)
    self._dataset = self._dataset.repeat()

    if self.params['mode'] != 'infer':
      self._dataset = self._dataset.map(
        lambda line: tf.py_func(
          self._parse_audio_transcript_element,
          [line],
          [tf.int32, tf.int32, self.params['dtype'], self.params['dtype'], tf.int32],
          stateful=False,
        ),
        num_parallel_calls=8,
      )
      self._dataset = self._dataset.padded_batch(
        self.params['batch_size'],
        padded_shapes=([None], 1, [None, self.params['num_audio_features']], [None], 1),
        padding_values=(0,0,tf.cast(0.,dtype=self.params['dtype']),
          tf.cast(0.,dtype=self.params['dtype']),0)
      )
    else:
      self._dataset = self._dataset.map(
        lambda line: tf.py_func(
          self._parse_transcript_element,
          [line],
          [tf.int32, tf.int32],
          stateful=False,
        ),
        num_parallel_calls=8,
      )
      self._dataset = self._dataset.padded_batch(
        self.params['batch_size'],
        padded_shapes=([None], 1)
    )

    self._iterator = self._dataset.prefetch(8).make_initializable_iterator()

    if self.params['mode'] != 'infer':
      text, text_length, spec, stop_token_target, spec_length = self._iterator.get_next()
      # need to explicitly set batch size dimension
      # (it is employed in the model)
      spec.set_shape([self.params['batch_size'], None,
                 self.params['num_audio_features']])
      stop_token_target.set_shape([self.params['batch_size'], None])
      spec_length = tf.reshape(spec_length, [self.params['batch_size']])
    else:
      text, text_length = self._iterator.get_next()
    text.set_shape([self.params['batch_size'], None])
    text_length = tf.reshape(text_length, [self.params['batch_size']])

    self._input_tensors = {}
    self._input_tensors["source_tensors"] = [text, text_length]
    if self.params['mode'] != 'infer':
      self._input_tensors['target_tensors'] = [spec, stop_token_target, spec_length]

  def _parse_audio_transcript_element(self, element):
    """Parses tf.data element from TextLineDataset into audio and text.

    Args:
      element: tf.data element from TextLineDataset.

    Returns:
      tuple: text_input text as `np.array` of ids, text_input text length,
      source audio features as `np.array`, stop token targets as `np.array`,
      length of source sequence,
      .
    """
    audio_filename, transcript = element
    if not six.PY2:
      transcript = str(transcript, 'utf-8')
    transcript = transcript.lower()
    text_input = np.array([self.params['char2idx'][c] for c in unicode(transcript,"utf-8")])
    if self.params.get("pad_EOS", True):
      text_input = np.append(text_input, self.params['char2idx']["~"])
    pad_to = self.params.get('pad_to', 8)
    if self.load_from_disk:
      file_path = os.path.join(self.params['dataset_location'],audio_filename+".npy")
      spectrogram = np.load(file_path)
      mag_power = self.params.get('mag_power', 2)
      if self.mel:
        spectrogram = get_mel(spectrogram, power=mag_power,
                              feature_normalize=self.params["feature_normalize"],
                              mean=self.params.get("feature_normalize_mean", 0.),
                              std=self.params.get("feature_normalize_std", 1.),
                              mel_basis=self.mel_basis,
                              n_mels=self.params['num_audio_features']
                             )
      else:
        if mag_power != 1:
          spectrogram = spectrogram * mag_power
          spectrogram = np.clip(spectrogram, a_min=np.log(1e-5), a_max=None)
        # Else it is a magnitude spec, and we need to normalize
        if self.params["feature_normalize"]:
          spectrogram = normalize(spectrogram,
                                mean=self.params.get("feature_normalize_mean", 0.),
                                std=self.params.get("feature_normalize_std", 0.))
    else:
      file_path = os.path.join(self.params['dataset_location'],audio_filename+".wav")
      spectrogram = get_speech_features_from_file(
        file_path, self.params['num_audio_features'], pad_to,
        features_type=self.params['output_type'],
        mag_power=self.params.get('mag_power', 2),
        feature_normalize=self.params["feature_normalize"],
        mean=self.params.get("feature_normalize_mean", 0.),
        std=self.params.get("feature_normalize_std", 1.)
      )
    if self.params.get("pad_EOS", True):
      spectrogram = np.pad(spectrogram, ((0,1),(0,0)), "constant", constant_values=0)
    stop_token_target = np.zeros([len(spectrogram)], dtype=self.params['dtype'].as_numpy_dtype())
    stop_token_target[-1] = 1.
    return np.int32(text_input), \
           np.int32([len(text_input)]), \
           spectrogram.astype(self.params['dtype'].as_numpy_dtype()), \
           stop_token_target.astype(self.params['dtype'].as_numpy_dtype()), \
           np.int32([len(spectrogram)])

  def _parse_transcript_element(self, transcript):
    """Parses text from file and returns array of text features.

    Args:
      text: the string to parse.

    Returns:
      tuple: target text as `np.array` of ids, target text length.
    """
    if not six.PY2:
      transcript = str(transcript, 'utf-8')
    transcript = transcript.lower()
    text_input = np.array([self.params['char2idx'][c] for c in unicode(transcript,"utf-8")])
    if self.params.get("pad_EOS", True):
      text_input = np.append(text_input, self.params['char2idx']["~"])

    return np.int32(text_input), \
           np.int32([len(text_input)])

  @property
  def input_tensors(self):
    return self._input_tensors

  def get_size_in_samples(self):
    """Returns the number of audio files."""
    return len(self._files)

  def get_magnitude_spec(self, spectrogram):
    """Returns an energy magnitude spectrogram. The processing depends on the 
    data leyer params.

    Args:
      spectrogram: output spec from model

    Returns:
      mag_spec: mag spec
    """
    spectrogram = spectrogram.astype(float)
    if self.mel:
      return inverse_mel(spectrogram, 
                         n_mels=self.params['num_audio_features'],
                         power=self.params.get('mag_power', 2),
                         feature_normalize=self.params["feature_normalize"],
                         mean=self.params.get("feature_normalize_mean", 0.),
                         std=self.params.get("feature_normalize_std", 1.),
                         mel_basis=self.mel_basis)
    else:
      if self.params["feature_normalize"]:
        spectrogram = self._denormalize(spectrogram)
      spectrogram = spectrogram * 1.0/self.params.get('mag_power', 2)
      mag_spec = np.exp(spectrogram)
      return mag_spec

  def _denormalize(self, spectrogram):
    return denormalize(spectrogram,
                       mean=self.params.get("feature_normalize_mean", 0.),
                       std=self.params.get("feature_normalize_std", 1.))
