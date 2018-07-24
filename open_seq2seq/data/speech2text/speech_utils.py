# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import math

import numpy as np
import python_speech_features as psf
import resampy as rs
import scipy.io.wavfile as wave


def get_speech_features_from_file(filename, num_features, pad_to=8,
                                  features_type='spectrogram',
                                  window_size=20e-3,
                                  window_stride=10e-3,
                                  augmentation=None):
  """Function to convert audio file to numpy array of features.

  Args:
    filename (string): WAVE filename.
    num_features (int): number of speech features in frequency domain.
    features_type (string): 'mfcc' or 'spectrogram'.
    window_size (float): size of analysis window in milli-seconds.
    window_stride (float): stride of analysis window in milli-seconds.
    augmentation (dict, optional): None or dictionary of augmentation parameters.
        If not None, has to have 'time_stretch_ratio',
        'noise_level_min', 'noise_level_max' fields, e.g.::
          augmentation={
            'time_stretch_ratio': 0.2,
            'noise_level_min': -90,
            'noise_level_max': -46,
          }
  Returns:
    np.array: np.array of audio features with shape=[num_time_steps, num_features].
  """
  # load audio signal
  fs, signal = wave.read(filename)
  return get_speech_features(
      signal, fs, num_features, pad_to, features_type,
      window_size, window_stride, augmentation,
  )


def normalize_signal(signal):
  """
  Normalize float32 signal to [-1, 1] range
  """
  return signal / np.max(np.abs(signal))


def augment_audio_signal(signal, fs, augmentation):
  """Function that performs audio signal augmentation.

  Args:
    signal (np.array): np.array containing raw audio signal.
    fs (float): frames per second.
    augmentation (dict): dictionary of augmentation parameters. See
        :func:`get_speech_features_from_file` for specification and example.
  Returns:
    np.array: np.array with augmented audio signal.
  """
  signal_float = normalize_signal(signal.astype(np.float32))

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

  return (normalize_signal(signal_float) * 32767.0).astype(np.int16)


def get_speech_features(signal, fs, num_features, pad_to=8,
                        features_type='spectrogram',
                        window_size=20e-3,
                        window_stride=10e-3,
                        augmentation=None):
  """Function to convert raw audio signal to numpy array of features.

  Args:
    signal (np.array): np.array containing raw audio signal.
    fs (float): frames per second.
    num_features (int): number of speech features in frequency domain.
    pad_to (int): if specified, the length will be padded to become divisible
        by ``pad_to`` parameter.
    features_type (string): 'mfcc' or 'spectrogram'.
    window_size (float): size of analysis window in milli-seconds.
    window_stride (float): stride of analysis window in milli-seconds.
    augmentation (dict, optional): dictionary of augmentation parameters. See
        :func:`get_speech_features_from_file` for specification and example.

  Returns:
    np.array: np.array of audio features with shape=[num_time_steps, num_features].
    audio_duration (float): duration of the signal in seconds
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
  else:
    signal = (normalize_signal(signal.astype(np.float32)) * 32767.0).astype(np.int16)

  audio_duration = len(signal) * 1.0/fs

  n_window_size = int(fs * window_size)
  n_window_stride = int(fs * window_stride)

  # making sure length of the audio is divisible by 8 (fp16 optimization)
  length = 1 + int(math.ceil(
      (1.0 * signal.shape[0] - n_window_size) / n_window_stride
  ))
  if pad_to > 0:
    if length % pad_to != 0:
      pad_size = (pad_to - length % pad_to) * n_window_stride
      signal = np.pad(signal, (0, pad_size), mode='constant')

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

  elif features_type == 'logfbank':
    features = psf.logfbank(signal=signal,
                            samplerate=fs,
                            winlen=window_size,
                            winstep=window_stride,
                            nfilt=num_features,
                            nfft=512,
                            lowfreq=0, highfreq=fs/2,
                            preemph=0.97)

  else:
    raise ValueError('Unknown features type: {}'.format(features_type))

  if pad_to > 0:
    assert features.shape[0] % pad_to == 0
  m = np.mean(features)
  s = np.std(features)
  features = (features - m) / s
  return features, audio_duration
