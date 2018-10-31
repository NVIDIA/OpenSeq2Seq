# Copyright (c) 2018 NVIDIA Corporation
import os
import six
import numpy as np
import tensorflow as tf
import pandas as pd

from open_seq2seq.data.data_layer import DataLayer
from open_seq2seq.data.text2speech.speech_utils import \
  get_speech_features_from_file

class WavenetDataLayer(DataLayer):
  """ Text to speech data layer class for Wavenet """

  @staticmethod
  def get_required_params():
    return dict(
        DataLayer.get_required_params(), **{
            "num_audio_features": int,
            "dataset_files": list
        }
    )

  @staticmethod
  def get_optional_params():
    return dict(
        DataLayer.get_optional_params(), **{
            "dataset_location": str
        }
    )

  def __init__(self, params, model, num_workers=None, worker_id=None):
    """
    Wavenet data layer constructor.

    See parent class for arguments description.

    Config parameters:

    * **num_audio_features** (int) --- number of spectrogram audio features
    * **dataset_files** (list) --- list with paths to all dataset .csv files

    * **dataset_location** (str) --- string with path to directory where wavs
      are stored
    """

    super(WavenetDataLayer, self).__init__(
        params,
        model,
        num_workers,
        worker_id
    )

    if self.params.get("dataset_location", None) is None:
      raise ValueError(
          "dataset_location must be specified when using LJSpeech"
      )

    names = ["wav_filename", "raw_transcript", "transcript"]
    sep = "\x7c"
    header = None

    self.sampling_rate = 22050
    self.n_fft = 1024

    self._files = None
    for csvs in params["dataset_files"]:
      files = pd.read_csv(
          csvs,
          encoding="utf-8",
          sep=sep,
          header=header,
          names=names,
          quoting=3
      )

      if self._files is None:
        self._files = files
      else:
        self._files = self._files.append(files)

    cols = "wav_filename"
    if self._files is not None:
      all_files = self._files.loc[:, cols].values
      self._files = self.split_data(all_files)

    self._size = self.get_size_in_samples()
    self._dataset = None
    self._iterator = None
    self._input_tensors = None

  @property
  def input_tensors(self):
    return self._input_tensors

  def get_size_in_samples(self):
    if self._files is not None:
      return len(self._files)
    else:
      return 0

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
  def iterator(self):
    return self._iterator

  def _parse_audio_element(self, element):
    """Parses tf.data element from TextLineDataset into audio."""
    audio_filename = element

    if six.PY2:
      audio_filename = unicode(audio_filename, "utf-8")
    else:
      audio_filename = str(audio_filename, "utf-8")

    file_path = os.path.join(
        self.params["dataset_location"],
        audio_filename + ".wav"
    )

    audio, spectrogram = get_speech_features_from_file(
        file_path,
        self.params["num_audio_features"],
        features_type="mel",
        data_min=1e-5,
        return_raw_audio=True
    )

    spectrogram = np.pad(
        spectrogram,
        ((0, 1), (0, 0)),
        "constant",
        constant_values=1e-5
    )
    assert len(audio) < len(spectrogram)*256, \
        "audio len: {}, spec*256 len: {}".format(len(audio), \
        len(spectrogram)*256)
    num_pad = len(spectrogram)*256 - len(audio)
    audio = np.pad(
        audio,
        (0, num_pad),
        "constant",
        constant_values=0
    )

    # upsample the spectrogram to match source length by repeating each value
    spectrogram = np.repeat(spectrogram, 256, axis=0)

    return audio.astype(self.params["dtype"].as_numpy_dtype()), \
      np.int32([len(audio)]), \
      spectrogram.astype(self.params["dtype"].as_numpy_dtype()), \
      np.int32([len(spectrogram)])

  def _parse_spectrogram_element(self, element):
    audio, au_length, spectrogram, spec_length = \
      self._parse_audio_element(element)
    return spectrogram, spec_length

  def create_interactive_placeholders(self):
    self._source = tf.placeholder(
        dtype=self.params["dtype"],
        shape=[self.params["batch_size"], None]
    )
    self._src_length = tf.placeholder(
        dtype=tf.int32,
        shape=[self.params["batch_size"]]
    )

    self._spec = tf.placeholder(
        dtype=self.params["dtype"],
        shape=[self.params["batch_size"], None,
               self.params["num_audio_features"]]
    )
    self._spec_length = tf.placeholder(
        dtype=tf.int32,
        shape=[self.params["batch_size"]]
    )
    self._spec_offset = tf.placeholder(
        dtype=tf.int32,
        shape=()
    )

    self._input_tensors = {}
    self._input_tensors["source_tensors"] = [
        self._source, self._src_length, self._spec, self._spec_length,
        self._spec_offset
    ]

  def create_feed_dict(self, model_in):
    """
    Creates the feed dict for interactive infer using a spectrogram

    Args:
      model_in: tuple(
        source: source audio
        src_length: length of the source
        spec: conditioning spectrogram
        spec_length: length of the spectrogram
        spec_offset: iterative index for position of receptive field window
      )
    """

    source, src_length, spec, spec_length, spec_offset = model_in

    return {
        self._source: source,
        self._src_length: src_length,
        self._spec: spec,
        self._spec_length: spec_length,
        self._spec_offset: spec_offset
    }

  def build_graph(self):
    """ builds data reading graph """
    self._dataset = tf.data.Dataset.from_tensor_slices(self._files)

    if self.params["shuffle"]:
      self._dataset = self._dataset.shuffle(self._size)
    self._dataset = self._dataset.repeat()

    num_audio_features = self.params["num_audio_features"]

    if self.params["mode"] != "infer":
      self._dataset = self._dataset.map(
          lambda line: tf.py_func(
              self._parse_audio_element,
              [line],
              [self.params["dtype"], tf.int32, self.params["dtype"], tf.int32],
              stateful=False
          ),
          num_parallel_calls=8
      )

      self._dataset = self._dataset.padded_batch(
          self.params["batch_size"],
          padded_shapes=([None], 1, [None, num_audio_features], 1)
      )

    else:
      raise ValueError("Non-interactive infer is not supported")

    self._iterator = self._dataset.prefetch(tf.contrib.data.AUTOTUNE) \
      .make_initializable_iterator()

    if self.params["mode"] != "infer":
      source, src_length, spec, spec_length = self._iterator.get_next()
      spec.set_shape([self.params["batch_size"], None, num_audio_features])
      spec_length = tf.reshape(spec_length, [self.params["batch_size"]])

      source.set_shape([self.params["batch_size"], None])
      src_length = tf.reshape(src_length, [self.params["batch_size"]])

      self._input_tensors = {}
      self._input_tensors["source_tensors"] = [
          source, src_length, spec, spec_length
      ]
      self._input_tensors["target_tensors"] = [source, src_length]

    else:
      raise ValueError("Non-interactive infer is not supported")
