# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import scipy.io.wavfile as wave
import resampy as rs
import python_speech_features as psf
import numpy as np
import librosa
import librosa.filters
import math


def get_speech_features_from_file(filename, num_features, pad_to=8,
                                  features_type='spectrogram',
                                  window_size=50e-3,
                                  window_stride=125e-4,
                                  augmentation=None,
                                  mag_power=2,
                                  feature_normalize=True):
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
  if features_type == "mel" or features_type =="test" or features_type=='spectrogram':
    signal, fs = librosa.core.load(filename, sr=None)
  else:
    fs, signal = wave.read(filename)
  return get_speech_features(
    signal, fs, num_features, pad_to, features_type,
    window_size, window_stride, augmentation, mag_power, feature_normalize
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


def get_speech_features(signal, fs, num_features, pad_to=8,
                        features_type='spectrogram',
                        window_size=50e-3,
                        window_stride=125e-4,
                        augmentation=None,
                        mag_power=2, 
                        feature_normalize=True):
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
  if augmentation is not None and features_type is not 'mel':
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

  # n_window_size = float(fs * window_size)
  # n_window_stride = float(fs * window_stride)
  # n_window_size = int(fs * window_size)
  # n_window_stride = int(fs * window_stride)
  n_window_size = 1024
  n_window_stride = 256

  # making sure length of the audio is divisible by 8 (fp16 optimization)
  # length = 1 + float(math.ceil(
  #   (1.0 * signal.shape[0] - n_window_size) / n_window_stride)
  # )
  length = 1 + int(math.ceil(
    (1.0 * signal.shape[0] - n_window_size) / n_window_stride)
  )
  if pad_to > 0:
    # if int(length) % pad_to != 0:
    if length % pad_to != 0:
      # pad_size = int(math.ceil((pad_to - length % pad_to) * n_window_stride))
      pad_size = (pad_to - length % pad_to) * n_window_stride
      signal = np.pad(signal, (0, pad_size), mode='reflect')


  if features_type == 'spectrogram':
    complex_spec = librosa.stft(y=signal,
                                n_fft=n_window_size)
    mag, _ = librosa.magphase(complex_spec)
    features = np.log(np.clip(mag, a_min=1e-5, a_max=None)).T
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
                        nfft=2048,
                        lowfreq=125, highfreq=7600,
                        preemph=0.97,
                        ceplifter=2*num_features,
                        appendEnergy=False,
                        winfunc=np.hanning)
  elif features_type == 'mel':
    features = librosa.feature.melspectrogram(y=signal,
                                      sr=fs,
                                      n_fft=n_window_size,
                                      hop_length=n_window_stride,
                                      n_mels=num_features,
                                      power=mag_power)
    features = np.log(np.clip(features, a_min=1e-5, a_max=None)).T
    # features = features.T
  elif features_type == 'test':
    n_window_size = 512
    complex_spec = librosa.stft(y=signal,
                                n_fft=n_window_size)
    assert num_features <= n_window_size // 2 + 1, \
        "num_features for spectrogram should be <= (fs * window_size // 2 + 1)"
    features = np.concatenate((complex_spec.real[:num_features,:],complex_spec.imag[:num_features,:]), axis=0)
    features = np.log(np.clip(features, a_min=1e-5, a_max=None)).T
  else:
    raise ValueError('Unknown features type: {}'.format(features_type))

  # print(n_window_size)
  # print(n_window_stride)
  # print(length)
  # print(signal.shape)
  # if pad_to > 0:
  #   if length % pad_to != 0:
  #     print(pad_size)
  #     print(new_signal.shape)
  # print(features.shape)

  # assert features.shape[0] % pad_to == 0
  if feature_normalize:
    m = np.mean(features)
    s = np.std(features)
    features = (features - m) / s
  return features

def inverse_mel(mel_spec, fs=22050, n_fft=1024, n_mels=80):
  mel_basis = librosa.filters.mel(fs, n_fft, n_mels=n_mels)
  # print(mel_spec.shape)
  # print(mel_basis.shape)
  # spec = np.dot(mel_spec, mel_basis)
  # print(spec.shape)
  mel_spec = np.exp(mel_spec)
  return np.dot(mel_spec, mel_basis)
