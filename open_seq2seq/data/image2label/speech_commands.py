import os
import six
import numpy as np
import tensorflow as tf
import pandas as pd
import librosa

from open_seq2seq.data.data_layer import DataLayer
from open_seq2seq.data.text2speech.speech_utils import \
  get_speech_features_from_file

class SpeechCommandsDataLayer(DataLayer):

  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), ** {
        "dataset_files": list,
        "dataset_location": str,
        "num_audio_features": int,
        "num_labels": int
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
        "cache_data": bool,
        "augment_data": bool
    })

  def split_data(self, data):
    if self.params["mode"] != "train" and self._num_workers is not None:
      size = len(data)
      start = size // self._num_workers * self._worker_id

      if self._worker_id == self._num_workers - 1:
        end = size
      else:
        end = size // self._num_workers * (self._worker_id + 1)

      return data[start:end]

    return data

  @property
  def input_tensors(self):
    return self._input_tensors

  @property
  def iterator(self):
    return self._iterator

  def get_size_in_samples(self):
    if self._files is not None:
      return len(self._files)
    else:
      return 0

  def __init__(self, params, model, num_workers=None, worker_id=None):
    """
    ResNet Speech Commands data layer constructor.

    Config parameters:

    * **dataset_files** (list) --- list with paths to all dataset .csv files
    * **dataset_location** (str) --- string with path to directory where .wavs
      are stored
    * **num_audio_features** (int) --- number of spectrogram audio features and 
      image dimension
    * **num_labels** (int) --- number of classes in dataset
    
    * **cache_data** (bool) --- cache the training data in the first epoch
    * **augment_data** (bool) --- add time stretch and noise to training data
    """

    super(SpeechCommandsDataLayer, self).__init__(params, model, num_workers, worker_id)

    if self.params["mode"] == "infer":
      raise ValueError("Inference is not supported on SpeechCommandsDataLayer")

    self._files = None
    for file in self.params["dataset_files"]:
      csv_file = pd.read_csv(
        os.path.join(self.params["dataset_location"], file),
        encoding="utf-8",
        sep=",",
        header=None,
        names=["label", "wav_filename"],
        dtype=str
      )

    if self._files is None:
      self._files = csv_file
    else:
      self._files.append(csv_file)

    cols = ["label", "wav_filename"]

    if self._files is not None:
      all_files = self._files.loc[:, cols].values
      self._files = self.split_data(all_files)

    self._size = self.get_size_in_samples()
    self._iterator = None
    self._input_tensors = None

  def preprocess_image(self, image):
    """Crops or pads a spectrogram into a fixed dimension square image
    """
    dim = self.params["num_audio_features"]

    if image.shape[0] > dim: # randomly slice to square
      offset = np.random.randint(0, image.shape[0] - dim + 1)
      image = image[offset:offset + dim, :]

    else: # symmetrically pad with zeros to square
      pad_left = (dim - image.shape[0]) // 2
      pad_right = (dim - image.shape[0]) // 2

      if (dim - image.shape[0]) % 2 == 1:
        pad_right += 1

      image = np.pad(
          image, 
          ((pad_left, pad_right), (0, 0)), 
          "constant"
      )

    assert image.shape == (dim, dim)

    # add dummy dimension as channels
    image = np.expand_dims(image, -1)

    return image

  def parse_element(self, element):
    """Reads an audio file and returns the augmented spectrogram image
    """
    label, audio_filename = element

    if six.PY2:
      audio_filename = unicode(audio_filename, "utf-8")
    else:
      audio_filename = str(audio_filename, "utf-8")

    file_path = os.path.join(
        self.params["dataset_location"],
        audio_filename
    )

    if self.params["mode"] == "train" and self.params.get("augment_data", False):
      augmentation = { 
        "time_stretch_ratio": 0.2,
        "noise_level_min": -90,
        "noise_level_max": -46,
      }
    else:
      augmentation = None

    spectrogram = get_speech_features_from_file(
        file_path,
        self.params["num_audio_features"],
        features_type="mel",
        data_min=1e-5,
        augmentation=augmentation
    )

    image = self.preprocess_image(spectrogram)

    return np.float32(image), np.int32(label)

  def build_graph(self):
    dataset = tf.data.Dataset.from_tensor_slices(self._files)

    cache_data = self.params.get("cache_data", False)

    if not cache_data:
      if self.params["shuffle"]:
        dataset = dataset.shuffle(self._size)

    dataset = dataset.map(
        lambda line: tf.py_func(
            self.parse_element,
            [line],
            [tf.float32, tf.int32],
            stateful=False
        ),
        num_parallel_calls=8
    )

    if cache_data:
      dataset = dataset.cache()
      if self.params["shuffle"]:
        dataset = dataset.shuffle(self._size)  

    if self.params["repeat"]:
      dataset = dataset.repeat()

    dataset = dataset.batch(self.params["batch_size"])
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    self._iterator = dataset.make_initializable_iterator()
    inputs, labels = self._iterator.get_next()

    inputs.set_shape([
        self.params["batch_size"], 
        self.params["num_audio_features"], 
        self.params["num_audio_features"], 
        1
    ])
    labels = tf.one_hot(labels, self.params["num_labels"])
    labels.set_shape([self.params["batch_size"], self.params["num_labels"]])

    if self.params["mode"] == "train":
      tf.summary.image("augmented_images", inputs, max_outputs=1)

    self._input_tensors = {
        "source_tensors": [inputs],
        "target_tensors": [labels]
    }