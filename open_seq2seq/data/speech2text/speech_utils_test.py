# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import math
import os

import numpy as np
import numpy.testing as npt
import scipy.io.wavfile as wave
import tensorflow as tf
from six.moves import range

from .speech_utils import get_speech_features, get_speech_features_from_file, \
                          augment_audio_signal


class SpeechUtilsTests(tf.test.TestCase):

  def test_augment_audio_signal(self):
    filename = 'open_seq2seq/test_utils/toy_speech_data/wav_files/46gc040q.wav'
    freq_s, signal = wave.read(filename)
    augmentation = {
        'time_stretch_ratio': 0.2,
        'noise_level_min': -90,
        'noise_level_max': -46,
    }
    # just checking length requirements here for now
    for _ in range(100):
      signal_augm = augment_audio_signal(signal, freq_s, augmentation)
      self.assertLessEqual(signal.shape[0] * 0.8, signal_augm.shape[0])
      self.assertGreaterEqual(signal.shape[0] * 1.2, signal_augm.shape[0])
    augmentation = {
        'time_stretch_ratio': 0.5,
        'noise_level_min': -90,
        'noise_level_max': -46,
    }
    # just checking length requirements here for now
    for _ in range(100):
      signal_augm = augment_audio_signal(signal, freq_s, augmentation)
      self.assertLessEqual(signal.shape[0] * 0.5, signal_augm.shape[0])
      self.assertGreaterEqual(signal.shape[0] * 1.5, signal_augm.shape[0])

  def test_get_speech_features_from_file(self):
    dirname = 'open_seq2seq/test_utils/toy_speech_data/wav_files/'
    for name in ['46gc040q.wav', '206o0103.wav', '48rc041b.wav']:
      filename = os.path.join(dirname, name)
      for num_features in [161, 120]:
        for window_stride in [10e-3, 5e-3, 40e-3]:
          for window_size in [20e-3, 30e-3]:
            for features_type in ['spectrogram', 'mfcc', 'logfbank']:
              freq_s, signal = wave.read(filename)
              n_window_size = int(freq_s * window_size)
              n_window_stride = int(freq_s * window_stride)
              length = 1 + int(math.ceil(
                  (1.0 * signal.shape[0] - n_window_size) / n_window_stride
              ))
              if length % 8 != 0:
                length += 8 - length % 8
              right_shape = (length, num_features)
              input_features, _ = get_speech_features_from_file(
                  filename,
                  num_features,
                  features_type=features_type,
                  window_size=window_size,
                  window_stride=window_stride,
              )
              self.assertTrue(input_features.shape[0] % 8 == 0)

              self.assertTupleEqual(right_shape, input_features.shape)
              self.assertAlmostEqual(np.mean(input_features), 0.0)
              self.assertAlmostEqual(np.std(input_features), 1.0)
            # only for spectrogram
            with self.assertRaises(AssertionError):
              get_speech_features_from_file(
                  filename,
                  num_features=n_window_size // 2 + 2,
                  features_type='spectrogram',
                  window_size=window_size,
                  window_stride=window_stride,
              )

  def test_get_speech_features_from_file_augmentation(self):
    augmentation = {
        'time_stretch_ratio': 0.0,
        'noise_level_min': -90,
        'noise_level_max': -46,
    }
    filename = 'open_seq2seq/test_utils/toy_speech_data/wav_files/46gc040q.wav'
    num_features = 161
    input_features_clean, _ = get_speech_features_from_file(
        filename, num_features, augmentation=None,
    )
    input_features_augm, _ = get_speech_features_from_file(
        filename, num_features, augmentation=augmentation,
    )
    # just checking that result is different with and without augmentation
    self.assertTrue(np.all(np.not_equal(input_features_clean,
                                        input_features_augm)))

    augmentation = {
        'time_stretch_ratio': 0.2,
        'noise_level_min': -90,
        'noise_level_max': -46,
    }
    input_features_augm, _ = get_speech_features_from_file(
        filename, num_features, augmentation=augmentation,
    )
    self.assertNotEqual(
        input_features_clean.shape[0],
        input_features_augm.shape[0],
    )
    self.assertEqual(
        input_features_clean.shape[1],
        input_features_augm.shape[1],
    )

  def test_get_speech_features_with_sine(self):
    freq_s = 16000.0
    t_s = np.arange(0, 0.5, 1.0 / freq_s)
    signal = np.sin(2 * np.pi * 4000 * t_s)
    features, _ = get_speech_features(signal, freq_s, 161, pad_to=0)
    npt.assert_allclose(
        np.abs(features - features[0]),
        np.zeros_like(features),
        atol=1e-6,
    )
    for i in range(80):
      npt.assert_allclose(features[:, 79 - i], features[:, 81 + i], atol=1e-6)
      self.assertGreater(features[0, 80 - i], features[0, 80 - i - 1])


if __name__ == '__main__':
  tf.test.main()
