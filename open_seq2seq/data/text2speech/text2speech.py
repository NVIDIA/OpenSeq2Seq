# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import os
import six
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd

from six import string_types

from open_seq2seq.data.data_layer import DataLayer
from open_seq2seq.data.utils import load_pre_existing_vocabulary
from .speech_utils import get_speech_features_from_file,\
                          inverse_mel, normalize, denormalize

class Text2SpeechDataLayer(DataLayer):
  """Text-to-speech data layer class
  """

  @staticmethod
  def get_required_params():
    return dict(
        DataLayer.get_required_params(), **{
            'dataset': ['LJ'],
            'num_audio_features': None,
            'output_type': ['magnitude', 'mel'],
            'vocab_file': str,
            'dataset_files': list,
            'feature_normalize': bool,
        }
    )

  @staticmethod
  def get_optional_params():
    return dict(
        DataLayer.get_optional_params(), **{
            'dataset_location': str,
            'pad_to': int,
            'mag_power': int,
            'pad_EOS': bool,
            'pad_value': float,
            'feature_normalize_mean': float,
            'feature_normalize_std': float,
            'trim': bool,
            'data_min': float,
            'duration_min': int,
            'duration_max': int,
            'mel_type': ['slaney', 'htk']
        }
    )

  def __init__(self, params, model, num_workers=None, worker_id=None):
    """Text-to-speech data layer constructor.

    See parent class for arguments description.

    Config parameters:

    * **dataset** (str) --- The dataset to use. Currently 'LJ' for the LJSpeech
      1.1 dataset is supported.
    * **num_audio_features** (int) --- number of audio features to extract.
    * **output_type** (str) --- could be either "magnitude", or "mel".
    * **vocab_file** (str) --- path to vocabulary file.
    * **dataset_files** (list) --- list with paths to all dataset .csv files.
      File is assumed to be separated by "|".
    * **dataset_location** (string) --- string with path to directory where wavs
      are stored.
    * **feature_normalize** (bool) --- whether to normlize the data with a
      preset mean and std
    * **feature_normalize_mean** (bool) --- used for feature normalize.
      Defaults to 0.
    * **feature_normalize_std** (bool) --- used for feature normalize.
      Defaults to 1.
    * **mag_power** (int) --- the power to which the magnitude spectrogram is
      scaled to. Defaults to 1.
      1 for energy spectrogram
      2 for power spectrogram
      Defaults to 2.
    * **pad_EOS** (bool) --- whether to apply EOS tokens to both the text and
      the speech signal. Will pad at least 1 token regardless of pad_to value.
      Defaults to True.
    * **pad_value** (float) --- The value we pad the spectrogram with. Defaults
      to np.log(data_min).
    * **pad_to** (int) --- we pad such that the resulting datapoint is a
      multiple of pad_to.
      Defaults to 8.
    * **trim** (bool) --- Whether to trim silence via librosa or not. Defaults
      to False.
    * **data_min** (float) --- min clip value prior to taking the log. Defaults
      to 1e-5. Please change to 1e-2 if using htk mels.
    * **duration_min** (int) --- Minimum duration in steps for speech signal.
      All signals less than this will be cut from the training set. Defaults to
      0.
    * **duration_max** (int) --- Maximum duration in steps for speech signal.
      All signals greater than this will be cut from the training set. Defaults 
      to 4000.
    * **mel_type** (str): One of ['slaney', 'htk']. Decides which algorithm to
      use to compute mel specs.
      Defaults to htk.

    """
    super(Text2SpeechDataLayer, self).__init__(
        params,
        model,
        num_workers,
        worker_id
    )

    if self.params.get('dataset_location', None) is None:
      raise ValueError(
          "dataset_location must be specified when using the LJSpeech",
          "dataset"
      )
    names = ['wav_filename', 'raw_transcript', 'transcript']
    sep = '\x7c'
    header = None

    self._sampling_rate = 22050
    self._n_fft = 1024

    # Character level vocab
    self.params['char2idx'] = load_pre_existing_vocabulary(
        self.params['vocab_file'],
        min_idx=3,
        read_chars=True,
    )
    # Add the pad, start, and end chars
    self.params['char2idx']['<p>'] = 0
    self.params['char2idx']['<s>'] = 1
    self.params['char2idx']['</s>'] = 2
    self.params['idx2char'] = {i: w for w, i in self.params['char2idx'].items()}
    self.params['src_vocab_size'] = len(self.params['char2idx'])

    n_mels = self.params['num_audio_features']

    if "mel" in self.params["output_type"]:
      self._mel = True
      htk = True
      norm = None
      if self.params.get('mel_type', 'htk') == 'slaney':
        htk = False
        norm = 1
      self._mel_basis = librosa.filters.mel(
          sr=self._sampling_rate,
          n_fft=self._n_fft,
          n_mels=n_mels,
          htk=htk,
          norm=norm
      )
    else:
      self._mel = False
      self._mel_basis = None

    if self.params["interactive"]:
      return

    # Load csv files
    self._files = None
    for csvs in params['dataset_files']:
      files = pd.read_csv(
          csvs,
          encoding='utf-8',
          sep=sep,
          header=header,
          names=names,
          quoting=3
      )
      if self._files is None:
        self._files = files
      else:
        self._files = self._files.append(files)

    if self.params['mode'] != 'infer':
      cols = ['wav_filename', 'transcript']
    else:
      cols = 'transcript'

    all_files = self._files.loc[:, cols].values
    self._files = self.split_data(all_files)

    self._size = self.get_size_in_samples()
    self._dataset = None
    self._iterator = None
    self._input_tensors = None

  def split_data(self, data):
    if self.params['mode'] != 'train' and self._num_workers is not None:
      #Decrease num_eval for dev, since most data is thrown out anyways
      if self.params['mode'] == 'eval':
        start = self._worker_id * self.params['batch_size']
        end = start+self.params['batch_size']
        return data[start:end]
      size = len(data)
      start = size // self._num_workers * self._worker_id
      if self._worker_id == self._num_workers - 1:
        end = size
      else:
        end = size // self._num_workers * (self._worker_id + 1)
      return data[start:end]
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

    num_audio_features = self.params['num_audio_features']

    if self.params['mode'] != 'infer':
      self._dataset = self._dataset.map(
          lambda line: tf.py_func(
              self._parse_audio_transcript_element,
              [line],
              [tf.int32, tf.int32, self.params['dtype'], self.params['dtype'],\
               tf.int32],
              stateful=False,
          ),
          num_parallel_calls=8,
      )
      if (self.params.get("duration_max", None) or
          self.params.get("duration_max", None)):
        self._dataset = self._dataset.filter(
            lambda txt, txt_len, spec, stop, spec_len:
                tf.logical_and(
                    tf.less_equal(
                        spec_len,
                        self.params.get("duration_max", 4000)
                    ),
                    tf.greater_equal(
                        spec_len,
                        self.params.get("duration_min", 0)
                    )
                )
        )
      default_pad_value = np.log(self.params.get("data_min", 1e-5))
      pad_value = self.params.get("pad_value", default_pad_value)
      if self.params["feature_normalize"]:
        pad_value = self._normalize(pad_value)
      self._dataset = self._dataset.padded_batch(
          self.params['batch_size'],
          padded_shapes=(
              [None], 1, [None, num_audio_features], [None], 1
          ),
          padding_values=(
              0, 0, tf.cast(pad_value, dtype=self.params['dtype']),
              tf.cast(1., dtype=self.params['dtype']), 0
          )
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
          self.params['batch_size'], padded_shapes=([None], 1)
      )

    self._iterator = self._dataset.prefetch(tf.contrib.data.AUTOTUNE)\
                                  .make_initializable_iterator()

    if self.params['mode'] != 'infer':
      text, text_length, spec, stop_token_target, spec_length = self._iterator\
                                                                    .get_next()
      # need to explicitly set batch size dimension
      # (it is employed in the model)
      spec.set_shape(
          [self.params['batch_size'], None, num_audio_features]
      )
      stop_token_target.set_shape([self.params['batch_size'], None])
      spec_length = tf.reshape(spec_length, [self.params['batch_size']])
    else:
      text, text_length = self._iterator.get_next()
    text.set_shape([self.params['batch_size'], None])
    text_length = tf.reshape(text_length, [self.params['batch_size']])

    self._input_tensors = {}
    self._input_tensors["source_tensors"] = [text, text_length]
    if self.params['mode'] != 'infer':
      self._input_tensors['target_tensors'] = [
          spec, stop_token_target, spec_length
      ]

  def _parse_audio_transcript_element(self, element):
    """Parses tf.data element from TextLineDataset into audio and text.

    Args:
      element: tf.data element from TextLineDataset.

    Returns:
      tuple: text_input text as `np.array` of ids, text_input length,
      target audio features as `np.array`, stop token targets as `np.array`,
      length of target sequence.

    """
    audio_filename, transcript = element
    transcript = transcript.lower()
    if six.PY2:
      audio_filename = unicode(audio_filename, "utf-8")
      transcript = unicode(transcript, "utf-8")
    else:
      audio_filename = str(audio_filename, "utf-8")
      transcript = str(transcript, "utf-8")
    text_input = np.array(
        [self.params['char2idx'][c] for c in transcript]
    )
    pad_to = self.params.get('pad_to', 8)
    if self.params.get("pad_EOS", True):
      num_pad = pad_to - ((len(text_input) + 2) % pad_to)
      text_input = np.pad(
          text_input, ((1, 1)),
          "constant",
          constant_values=(
              (self.params['char2idx']["<s>"], self.params['char2idx']["</s>"])
          )
      )
      text_input = np.pad(
          text_input, ((0, num_pad)),
          "constant",
          constant_values=self.params['char2idx']["<p>"]
      )


    file_path = os.path.join(
        self.params['dataset_location'], audio_filename + ".wav"
    )
    if self._mel:
      features_type = "mel_htk"
      if self.params.get('mel_type', 'htk') == 'slaney':
        features_type = "mel_slaney"
    else:
      features_type = self.params['output_type']

    spectrogram = get_speech_features_from_file(
        file_path,
        self.params['num_audio_features'],
        features_type=features_type,
        n_fft=self._n_fft,
        mag_power=self.params.get('mag_power', 2),
        feature_normalize=self.params["feature_normalize"],
        mean=self.params.get("feature_normalize_mean", 0.),
        std=self.params.get("feature_normalize_std", 1.),
        trim=self.params.get("trim", False),
        data_min=self.params.get("data_min", 1e-5)
    )
    stop_token_target = np.zeros(
        [len(spectrogram)], dtype=self.params['dtype'].as_numpy_dtype()
    )
    if self.params.get("pad_EOS", True):
      num_pad = pad_to - ((len(spectrogram) + 1) % pad_to) + 1

      data_min = self.params.get("data_min", 1e-5)

      pad_value = self.params.get("pad_value", np.log(data_min))
      if self.params["feature_normalize"]:
        pad_value = self._normalize(pad_value)

      spectrogram = np.pad(
          spectrogram,
          ((0, num_pad), (0, 0)),
          "constant",
          constant_values=pad_value
      )
      stop_token_target = np.pad(
          stop_token_target, ((0, num_pad)), "constant", constant_values=1
      )
    else:
      stop_token_target[-1] = 1.

    assert len(text_input) % pad_to == 0
    assert len(spectrogram) % pad_to == 0
    return np.int32(text_input), \
           np.int32([len(text_input)]), \
           spectrogram.astype(self.params['dtype'].as_numpy_dtype()), \
           stop_token_target.astype(self.params['dtype'].as_numpy_dtype()), \
           np.int32([len(spectrogram)])

  def _parse_transcript_element(self, transcript):
    """Parses text from file and returns array of text features.

    Args:
      transcript: the string to parse.

    Returns:
      tuple: target text as `np.array` of ids, target text length.
    """

    transcript = transcript.lower()
    if six.PY2:
      transcript = unicode(transcript, "utf-8")
    else:
      transcript = str(transcript, "utf-8")
    text_input = np.array(
        [self.params['char2idx'][c] for c in transcript]
    )
    pad_to = self.params.get('pad_to', 8)
    if self.params.get("pad_EOS", True):
      num_pad = pad_to - ((len(text_input) + 2) % pad_to)
      text_input = np.pad(
          text_input, ((1, 1)),
          "constant",
          constant_values=(
              (self.params['char2idx']["<s>"], self.params['char2idx']["</s>"])
          )
      )
      text_input = np.pad(
          text_input, ((0, num_pad)),
          "constant",
          constant_values=self.params['char2idx']["<p>"]
      )

    return np.int32(text_input), \
           np.int32([len(text_input)])

  def parse_text_output(self, text):
    text = "".join([self.params['idx2char'][k] for k in text])
    return text

  def create_interactive_placeholders(self):
    self._text = tf.placeholder(
        dtype=tf.int32,
        shape=[self.params["batch_size"], None]
    )
    self._text_length = tf.placeholder(
        dtype=tf.int32,
        shape=[self.params["batch_size"]]
    )

    self._input_tensors = {}
    self._input_tensors["source_tensors"] = [self._text, self._text_length]

  def create_feed_dict(self, model_in):
    """ Creates the feed dict for interactive infer

    Args:
      model_in (str): The string to be spoken.

    Returns:
      feed_dict (dict): Dictionary with values for the placeholders.
    """
    if not isinstance(model_in, string_types):
      raise ValueError(
          "Text2Speech's interactive inference mode only supports string.",
          "Got {}". format(type(model_in))
      )
    text, text_length = self._parse_transcript_element(model_in)

    text = np.reshape(text, [self.params["batch_size"], -1])
    text_length = np.reshape(text_length, [self.params["batch_size"]])

    feed_dict = {
        self._text: text,
        self._text_length: text_length,
    }
    return feed_dict

  @property
  def input_tensors(self):
    return self._input_tensors

  @property
  def sampling_rate(self):
    return self._sampling_rate

  @property
  def n_fft(self):
    return self._n_fft

  def get_size_in_samples(self):
    """Returns the number of audio files."""
    return len(self._files)

  def get_magnitude_spec(self, spectrogram):
    """Returns an energy magnitude spectrogram. The processing depends on the
    data layer params.

    Args:
      spectrogram: output spec from model

    Returns:
      mag_spec: mag spec
    """
    spectrogram = spectrogram.astype(float)
    if self._mel:
      htk = True
      norm = None
      if self.params.get('mel_type', 'htk') == 'slaney':
        htk = False
        norm = 1
      return inverse_mel(
          spectrogram,
          fs=self._sampling_rate,
          n_fft=self._n_fft,
          n_mels=self.params['num_audio_features'],
          power=self.params.get('mag_power', 2),
          feature_normalize=self.params["feature_normalize"],
          mean=self.params.get("feature_normalize_mean", 0.),
          std=self.params.get("feature_normalize_std", 1.),
          mel_basis=self._mel_basis,
          htk=htk,
          norm=norm
      )
    # Else it is a mag spec
    else:
      if self.params["feature_normalize"]:
        spectrogram = self._denormalize(spectrogram)
      # Ensure that num_features is consistent with n_fft
      if self.params['num_audio_features'] < self._n_fft // 2 + 1:
        num_pad = (self._n_fft // 2 + 1) - spectrogram.shape[1]
        spectrogram = np.pad(
            spectrogram,
            ((0, 0), (0, num_pad)),
            "constant",
            constant_values=np.log(self.params.get("data_min", 1e-5))
        )
      spectrogram = spectrogram * 1.0 / self.params.get('mag_power', 2)
      mag_spec = np.exp(spectrogram)
      return mag_spec

  def _normalize(self, spectrogram):
    return normalize(
        spectrogram,
        mean=self.params.get("feature_normalize_mean", 0.),
        std=self.params.get("feature_normalize_std", 1.)
    )

  def _denormalize(self, spectrogram):
    return denormalize(
        spectrogram,
        mean=self.params.get("feature_normalize_mean", 0.),
        std=self.params.get("feature_normalize_std", 1.)
    )
