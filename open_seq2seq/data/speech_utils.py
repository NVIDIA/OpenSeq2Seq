# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import scipy.io.wavfile as wave
import resampy as rs
import python_speech_features as psf
import numpy as np
import math


def get_speech_features_from_file(filename, num_features,
                                  features_type='spectrogram',
                                  window_size=20e-3,
                                  window_stride=10e-3,
                                  augmentation=None):
  """
  :param filename: WAVE filename
  :param num_features: number of speech features in frequency domain
  :param features_type: 'mfcc' or 'spectrogram'
  :param window_size: size of analysis window, ms
  :param window_stride: stride of analysis window, ms
  :param augmentation: None or dictionary of augmentation parameters;
                       If not None, has to have 'time_stretch_ratio',
                       'noise_level_min', 'noise_level_max' fields, e.g.:
                       augmentation={'time_stretch_ratio': 0.2,
                                     'noise_level_min': -90,
                                     'noise_level_max': -46}
  :return: (num_time_steps, num_features) NumPy array
  """
  # load audio signal
  fs, signal = wave.read(filename)
  return get_speech_features(
    signal, fs, num_features, features_type,
    window_size, window_stride, augmentation,
  )


def augment_audio_signal(signal, fs, augmentation):
  """
  :param signal: np.array containing raw audio signal
  :param fs: float, frames per second
  :param augmentation: None or dictionary of augmentation parameters;
                       If not None, has to have 'time_stretch_ratio',
                       'noise_level_min', 'noise_level_max' fields, e.g.:
                       augmentation={'time_stretch_ratio': 0.2,
                                     'noise_level_min': -90,
                                     'noise_level_max': -46}
  :return: np.array with augmented signal
  """
  signal_float = signal.astype(np.float32) / 32768.0

  if augmentation['time_stretch_ratio'] > 0:
    # time stretch (might be slow)
    stretch_amount = 1.0 + (2.0 * np.random.rand() - 1.0) * \
                     augmentation['time_stretch_ratio']
    signal_float = rs.resample(
      signal_float,
      fs,
      int(fs * stretch_amount),
      filter='kaiser_fast',
    )

  # noise
  noise_level_db = np.random.randint(low=augmentation['noise_level_min'],
                                     high=augmentation['noise_level_max'])
  signal_float += np.random.randn(signal_float.shape[0]) * \
                  10.0 ** (noise_level_db / 20.0)

  return (signal_float * 32768.0).astype(np.int16)


def get_speech_features(signal, fs, num_features,
                        features_type='spectrogram',
                        window_size=20e-3,
                        window_stride=10e-3,
                        augmentation=None):
  """
  :param signal: np.array containing raw audio signal
  :param fs: float, frames per second
  :param num_features: number of speech features in frequency domain
  :param features_type: 'mfcc' or 'spectrogram'
  :param window_size: size of analysis window, ms
  :param window_stride: stride of analysis window, ms
  :param augmentation: None or dictionary of augmentation parameters;
                       If not None, has to have 'time_stretch_ratio',
                       'noise_level_min', 'noise_level_max' fields, e.g.:
                       augmentation={'time_stretch_ratio': 0.2,
                                     'noise_level_min': -90,
                                     'noise_level_max': -46}
  :return: (num_time_steps, num_features) NumPy array
  """
  if augmentation is not None:
    if 'time_stretch_ratio' not in augmentation:
      raise ValueError('time_stretch_ratio has to be included in augmentation '
                       'when augmentation it is not None')
    if 'noise_level_min' not in augmentation:
      raise ValueError('noise_level_min has to be included in augmentation '
                       'when augmentation it is not None')
    if 'noise_level_max' not in augmentation:
      raise ValueError('noise_level_max has to be included in augmentation '
                       'when augmentation it is not None')
    signal = augment_audio_signal(signal, fs, augmentation)

  n_window_size = int(fs * window_size)
  n_window_stride = int(fs * window_stride)

  # making sure length of the audio is divisible by 8 (fp16 optimization)
  length = 1 + int(math.ceil(
    (1.0 * signal.shape[0] - n_window_size) / n_window_stride)
  )
  if length % 8 != 0:
    pad_size = (8 - length % 8) * n_window_stride
    signal = np.pad(signal, (0, pad_size), mode='reflect')

  if features_type == 'spectrogram':
    frames = psf.sigproc.framesig(sig=signal,
                                  frame_len=n_window_size,
                                  frame_step=n_window_stride,
                                  winfunc=np.hanning)

    # features = np.log1p(psf.sigproc.powspec(frames, NFFT=N_window_size))
    features = psf.sigproc.logpowspec(frames, NFFT=n_window_size)
    assert num_features <= n_window_size // 2 + 1, \
        "num_features for spectrogram should be <= (fs * window_size // 2 + 1)"

    # cut high frequency part
    features = features[:, :num_features]

  elif features_type == 'mfcc':
    features = psf.mfcc(signal=signal,
                        samplerate=fs,
                        winlen=window_size,
                        winstep=window_stride,
                        numcep=num_features,
                        nfilt=2*num_features,
                        nfft=512,
                        lowfreq=0, highfreq=None,
                        preemph=0.97,
                        ceplifter=2*num_features,
                        appendEnergy=False)
  else:
    raise ValueError('Unknown features type: {}'.format(features_type))

  assert features.shape[0] % 8 == 0
  m = np.mean(features)
  s = np.std(features)
  features = (features - m) / s
  return features
