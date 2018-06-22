# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import numpy as np
import librosa
import librosa.filters
import math


def get_speech_features_from_file(filename, num_features, pad_to=8,
                                  features_type='magnitude',
                                  window_size=1024,
                                  window_stride=256,
                                  mag_power=2,
                                  feature_normalize=False,
                                  mean = 0,
                                  std = 1):
  """
  :param filename: WAVE filename
  :param num_features: number of speech features in frequency domain
  :param features_type: 'mel' or 'magnitude'
  :param window_size: size of analysis window, in samples
  :param window_stride: stride of analysis window, in samples
  :param mag_power: the power to which the magnitude spectrogram is scaled to
      1 for energy spectrogram
      2 for power spectrogram
      Defaults to 2.
  :return: (num_time_steps, num_features) NumPy array
  """
  # load audio signal
  if features_type == "mel" or features_type=='magnitude':
    signal, fs = librosa.core.load(filename, sr=None)
  return get_speech_features(
    signal, fs, num_features, pad_to, features_type,
    window_size, window_stride, mag_power, 
    feature_normalize, mean, std
  )

def get_speech_features(signal, fs, num_features, pad_to=8,
                        features_type='magnitude',
                        n_window_size=1024,
                        n_window_stride=256,
                        mag_power=2, 
                        feature_normalize=False,
                        mean = 0,
                        std = 1):
  """
  :param signal: np.array containing raw audio signal
  :param fs: float, frames per second
  :param num_features: number of speech features in frequency domain
  :param features_type: 'mel' or 'spectrogram'
  :param n_window_size: size of analysis window, in samples
  :param n_window_stride: stride of analysis window, in samples
  :param mag_power: the power to which the magnitude spectrogram is scaled to
      1 for energy spectrogram
      2 for power spectrogram
      Defaults to 2.
  :return: (num_time_steps, num_features) NumPy array
  """
  # Padding and fp16 is not currently implemented
  # making sure length of the audio is divisible by 8 (fp16 optimization)
  # length = 1 + float(math.ceil(
  #   (1.0 * signal.shape[0] - n_window_size) / n_window_stride)
  # )
  # length = 1 + int(math.ceil(
    # (1.0 * signal.shape[0] - n_window_size) / n_window_stride)
  # )
  # if pad_to > 0:
    # if int(length) % pad_to != 0:
    # if length % pad_to != 0:
      # pad_size = int(math.ceil((pad_to - length % pad_to) * n_window_stride))
      # pad_size = (pad_to - length % pad_to) * n_window_stride
      # signal = np.pad(signal, (0, pad_size), mode='reflect')

  if features_type == 'magnitude':
    complex_spec = librosa.stft(y=signal,
                                n_fft=n_window_size)
    mag, _ = librosa.magphase(complex_spec, power=mag_power)
    features = np.log(np.clip(mag, a_min=1e-5, a_max=None)).T
    assert num_features <= n_window_size // 2 + 1, \
        "num_features for spectrogram should be <= (fs * window_size // 2 + 1)"

    # cut high frequency part
    features = features[:, :num_features]
  elif features_type == 'mel':
    features = librosa.feature.melspectrogram(y=signal,
                                      sr=fs,
                                      n_fft=n_window_size,
                                      hop_length=n_window_stride,
                                      n_mels=num_features,
                                      power=mag_power)
    features = np.log(np.clip(features, a_min=1e-5, a_max=None)).T
    # features = features.T
  else:
    raise ValueError('Unknown features type: {}'.format(features_type))

  if feature_normalize:
    features = normalize(features, mean, std)

  return features

def get_mel(log_mag_spec, fs=22050, n_fft=1024, n_mels=80, power=2.,
            feature_normalize=False, mean=0, std=1):
  """
  Method to get mel spectrograms from saved energy spectrograms
  """
  mel_basis = librosa.filters.mel(fs, n_fft, n_mels=n_mels)
  mag_spec = np.exp(log_mag_spec)
  mag_spec = np.power(mag_spec, power)
  mel_spec = np.dot(mag_spec, mel_basis.T)
  mel_spec = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))
  if feature_normalize:
    mel_spec = normalize(mel_spec, mean, std)
  return mel_spec

def inverse_mel(mel_spec, fs=22050, n_fft=1024, n_mels=80, power=2., 
               feature_normalize=False, mean=0, std=1):
  """
  Very hacky method to reconstruct mag spec from mel
  """
  mel_basis = librosa.filters.mel(fs, n_fft, n_mels=n_mels)
  mel_spec = np.exp(mel_spec)
  mag_spec = np.dot(mel_spec, mel_basis)
  mag_spec = np.power(mag_spec, 1./power)
  mag_spec = mag_spec * 32
  if feature_normalize:
    mag_spec = normalize(mag_spec, mean, std)
  return mag_spec

def normalize(features, mean, std):
  return (features - mean) / std

def denormalize(features, mean, std):
  return features * std + mean