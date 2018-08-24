# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import librosa
import librosa.filters

def get_speech_features_from_file(
    filename,
    num_features,
    features_type='magnitude',
    n_fft=1024,
    hop_length=None,
    mag_power=2,
    feature_normalize=False,
    mean=0.,
    std=1.,
    trim=False,
    data_min=1e-5
):
  """ Helper function to retrieve spectrograms from wav files

  Args:
    filename (string): WAVE filename.
    num_features (int): number of speech features in frequency domain.
    features_type (string): 'magnitude' or 'mel'.
    n_fft (int): size of analysis window in samples.
    hop_length (int): stride of analysis window in samples.
    mag_power (int): power to raise magnitude spectrograms (prior to dot product
      with mel basis)
      1 for energy spectrograms
      2 fot power spectrograms
    feature_normalize (bool): whether to normalize the data with mean and std
    mean (float): if normalize is enabled, the mean to normalize to
    std (float): if normalize is enabled, the deviation to normalize to
    trim (bool): Whether to trim silence via librosa or not
    data_min (float): min clip value prior to taking the log.

  Returns:
    np.array: np.array of audio features with shape=[num_time_steps,
    num_features].
  """
  # load audio signal
  signal, fs = librosa.core.load(filename, sr=None)
  if hop_length is None:
    hop_length = int(n_fft / 4)
  if trim:
    signal, _ = librosa.effects.trim(
        signal,
        frame_length=int(n_fft/2),
        hop_length=int(hop_length/2)
    )
  return get_speech_features(
      signal, fs, num_features, features_type, n_fft,
      hop_length, mag_power, feature_normalize, mean, std, data_min
  )


def get_speech_features(
    signal,
    fs,
    num_features,
    features_type='magnitude',
    n_fft=1024,
    hop_length=256,
    mag_power=2,
    feature_normalize=False,
    mean=0.,
    std=1.,
    data_min=1e-5
):
  """ Helper function to retrieve spectrograms from loaded wav

  Args:
    signal: signal loaded with librosa.
    fs (int): sampling frequency in Hz.
    num_features (int): number of speech features in frequency domain.
    features_type (string): 'magnitude' or 'mel'.
    n_fft (int): size of analysis window in samples.
    hop_length (int): stride of analysis window in samples.
    mag_power (int): power to raise magnitude spectrograms (prior to dot product
      with mel basis)
      1 for energy spectrograms
      2 fot power spectrograms
    feature_normalize(bool): whether to normalize the data with mean and std
    mean(float): if normalize is enabled, the mean to normalize to
    std(float): if normalize is enabled, the deviation to normalize to
    data_min (float): min clip value prior to taking the log.

  Returns:
    np.array: np.array of audio features with shape=[num_time_steps,
    num_features].
  """
  if features_type == 'magnitude':
    complex_spec = librosa.stft(y=signal, n_fft=n_fft)
    mag, _ = librosa.magphase(complex_spec, power=mag_power)
    features = np.log(np.clip(mag, a_min=data_min, a_max=None)).T
    assert num_features <= n_fft // 2 + 1, \
        "num_features for spectrogram should be <= (fs * window_size // 2 + 1)"

    # cut high frequency part
    features = features[:, :num_features]
  if 'mel' in features_type:
    htk = True
    norm = None
    if 'slaney' in features_type:
      htk = False
      norm = 1
    features = librosa.feature.melspectrogram(
        y=signal,
        sr=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=num_features,
        power=mag_power,
        htk=htk,
        norm=norm
    )
    features = np.log(np.clip(features, a_min=data_min, a_max=None)).T

  if feature_normalize:
    features = normalize(features, mean, std)

  return features


def get_mel(
    log_mag_spec,
    fs=22050,
    n_fft=1024,
    n_mels=80,
    power=2.,
    feature_normalize=False,
    mean=0,
    std=1,
    mel_basis=None,
    data_min=1e-5,
    htk=True,
    norm=None
):
  """
  Method to get mel spectrograms from magnitude spectrograms

  Args:
    log_mag_spec (np.array): log of the magnitude spec
    fs (int): sampling frequency in Hz
    n_fft (int): size of fft window in samples
    n_mels (int): number of mel features
    power (float): power of the mag spectrogram
    feature_normalize (bool): whether the mag spec was normalized
    mean (float): normalization param of mag spec
    std (float): normalization param of mag spec
    mel_basis (np.array): optional pre-computed mel basis to save computational
      time if passed. If not passed, it will call librosa to construct one
    data_min (float): min clip value prior to taking the log.
    htk (bool): whther to compute the mel spec with the htk or slaney algorithm
    norm: Should be None for htk, and 1 for slaney

  Returns:
    np.array: mel_spec with shape [time, n_mels]
  """
  if mel_basis is None:
    mel_basis = librosa.filters.mel(
        fs,
        n_fft,
        n_mels=n_mels,
        htk=htk,
        norm=norm
    )
  log_mag_spec = log_mag_spec * power
  mag_spec = np.exp(log_mag_spec)
  mel_spec = np.dot(mag_spec, mel_basis.T)
  mel_spec = np.log(np.clip(mel_spec, a_min=data_min, a_max=None))
  if feature_normalize:
    mel_spec = normalize(mel_spec, mean, std)
  return mel_spec


def inverse_mel(
    log_mel_spec,
    fs=22050,
    n_fft=1024,
    n_mels=80,
    power=2.,
    feature_normalize=False,
    mean=0,
    std=1,
    mel_basis=None,
    htk=True,
    norm=None
):
  """
  Reconstructs magnitude spectrogram from a mel spectrogram by multiplying it
  with the transposed mel basis.

  Args:
    log_mel_spec (np.array): log of the mel spec
    fs (int): sampling frequency in Hz
    n_fft (int): size of fft window in samples
    n_mels (int): number of mel features
    power (float): power of the mag spectrogram that was used to generate the
      mel spec
    feature_normalize (bool): whether the mel spec was normalized
    mean (float): normalization param of mel spec
    std (float): normalization param of mel spec
    mel_basis (np.array): optional pre-computed mel basis to save computational
      time if passed. If not passed, it will call librosa to construct one
    htk (bool): whther to compute the mel spec with the htk or slaney algorithm
    norm: Should be None for htk, and 1 for slaney

  Returns:
    np.array: mag_spec with shape [time, n_fft/2 + 1]
  """
  if mel_basis is None:
    mel_basis = librosa.filters.mel(
        fs,
        n_fft,
        n_mels=n_mels,
        htk=htk,
        norm=norm
    )
  if feature_normalize:
    log_mel_spec = denormalize(log_mel_spec, mean, std)
  mel_spec = np.exp(log_mel_spec)
  mag_spec = np.dot(mel_spec, mel_basis)
  mag_spec = np.power(mag_spec, 1. / power)
  return mag_spec


def normalize(features, mean, std):
  """
  Normalizes features with the specificed mean and std
  """
  return (features - mean) / std


def denormalize(features, mean, std):
  """
  Normalizes features with the specificed mean and std
  """
  return features * std + mean
