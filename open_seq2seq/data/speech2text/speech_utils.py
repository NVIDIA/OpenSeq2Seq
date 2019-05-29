# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import math
import os

import h5py
import numpy as np
import resampy as rs
import scipy.io.wavfile as wave
BACKENDS = []
try:
  import python_speech_features as psf
  BACKENDS.append('psf')
except ImportError:
  pass
try:
  import librosa
  BACKENDS.append('librosa')
except ImportError:
  pass

WINDOWS_FNS = {"hanning": np.hanning, "hamming": np.hamming, "none": None}


class PreprocessOnTheFlyException(Exception):
  """ Exception that is thrown to not load preprocessed features from disk;
  recompute on-the-fly.
  This saves disk space (if you're experimenting with data input
  formats/preprocessing) but can be slower.
  The slowdown is especially apparent for small, fast NNs."""
  pass


class RegenerateCacheException(Exception):
  """ Exception that is thrown to force recomputation of (preprocessed) features
  """
  pass


def load_features(path, data_format):
  """ Function to load (preprocessed) features from disk

  Args:
      :param path:    the path where the features are stored
      :param data_format:  the format in which the features are stored
      :return:        tuple of (features, duration)
      """
  if data_format == 'hdf5':
    with h5py.File(path + '.hdf5', "r") as hf5_file:
      features = hf5_file["features"][:]
      duration = hf5_file["features"].attrs["duration"]
  elif data_format == 'npy':
    features, duration = np.load(path + '.npy')
  elif data_format == 'npz':
    data = np.load(path + '.npz')
    features = data['features']
    duration = data['duration']
  else:
    raise ValueError("Invalid data format for caching: ", data_format, "!\n",
                     "options: hdf5, npy, npz")
  return features, duration


def save_features(features, duration, path, data_format, verbose=False):
  """ Function to save (preprocessed) features to disk

  Args:
      :param features:            features
      :param duration:            metadata: duration in seconds of audio file
      :param path:                path to store the data
      :param data_format:              format to store the data in ('npy',
      'npz',
      'hdf5')
  """
  if verbose: print("Saving to: ", path)

  if data_format == 'hdf5':
    with h5py.File(path + '.hdf5', "w") as hf5_file:
      dset = hf5_file.create_dataset("features", data=features)
      dset.attrs["duration"] = duration
  elif data_format == 'npy':
    np.save(path + '.npy', [features, duration])
  elif data_format == 'npz':
    np.savez(path + '.npz', features=features, duration=duration)
  else:
    raise ValueError("Invalid data format for caching: ", data_format, "!\n",
                     "options: hdf5, npy, npz")


def get_preprocessed_data_path(filename, params):
  """ Function to convert the audio path into the path to the preprocessed
  version of this audio
  Args:
      :param filename:    WAVE filename
      :param params:      dictionary containing preprocessing parameters
      :return:            path to new file (without extension). The path is
      generated from the relevant preprocessing parameters.
  """
  if isinstance(filename, bytes):  # convert binary string to normal string
    filename = filename.decode('ascii')

  filename = os.path.realpath(filename)  # decode symbolic links

  ## filter relevant parameters # TODO is there a cleaner way of doing this?
  # print(list(params.keys()))
  ignored_params = ["cache_features", "cache_format", "cache_regenerate",
                    "vocab_file", "dataset_files", "shuffle", "batch_size",
                    "max_duration",
                    "mode", "interactive", "autoregressive", "char2idx",
                    "tgt_vocab_size", "idx2char", "dtype"]

  def fix_kv(text):
    """ Helper function to shorten length of filenames to get around
    filesystem path length limitations"""
    text = str(text)
    text = text.replace("speed_perturbation_ratio", "sp") \
      .replace("noise_level_min", "nlmin", ) \
      .replace("noise_level_max", "nlmax") \
      .replace("add_derivatives", "d") \
      .replace("add_second_derivatives", "dd")
    return text

  # generate the identifier by simply concatenating preprocessing key-value
  # pairs as strings.
  preprocess_id = "-".join(
      [fix_kv(k) + "_" + fix_kv(v) for k, v in params.items() if
       k not in ignored_params])

  preprocessed_dir = os.path.dirname(filename).replace("wav",
                                                       "preprocessed-" +
                                                       preprocess_id)
  preprocessed_path = os.path.join(preprocessed_dir,
                                   os.path.basename(filename).replace(".wav",
                                                                      ""))

  # create dir if it doesn't exist yet
  if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

  return preprocessed_path


def get_speech_features_from_file(filename, params):
  """Function to get a numpy array of features, from an audio file.
      if params['cache_features']==True, try load preprocessed data from
      disk, or store after preprocesseng.
      else, perform preprocessing on-the-fly.

  Args:
    filename (string): WAVE filename.
    params (dict): the following parameters
      num_features (int): number of speech features in frequency domain.
      features_type (string): 'mfcc' or 'spectrogram'.
      window_size (float): size of analysis window in milli-seconds.
      window_stride (float): stride of analysis window in milli-seconds.
      augmentation (dict, optional): dictionary of augmentation parameters. See
        :func:`augment_audio_signal` for specification and example.
      window (str): window function to apply
      dither (float): weight of Gaussian noise to apply to input signal for
          dithering/preventing quantization noise
      num_fft (int): size of fft window to use if features require fft,
          defaults to smallest power of 2 larger than window size
      norm_per_feature (bool): if True, the output features will be normalized
          (whitened) individually. if False, a global mean/std over all features
          will be used for normalization
  Returns:
    np.array: np.array of audio features with shape=[num_time_steps,
    num_features].
  """
  cache_features = params.get('cache_features', False)
  cache_format = params.get('cache_format', 'hdf5')
  cache_regenerate = params.get('cache_regenerate', False)
  try:
    if not cache_features:
      raise PreprocessOnTheFlyException(
          "on-the-fly preprocessing enforced with 'cache_features'==True")

    if cache_regenerate:
      raise RegenerateCacheException("regenerating cache...")

    preprocessed_data_path = get_preprocessed_data_path(filename, params)
    features, duration = load_features(preprocessed_data_path,
                                       data_format=cache_format)

  except PreprocessOnTheFlyException:
    sample_freq, signal = wave.read(filename)
    features, duration = get_speech_features(signal, sample_freq, params)

  except (OSError, FileNotFoundError, RegenerateCacheException):
    sample_freq, signal = wave.read(filename)
    features, duration = get_speech_features(signal, sample_freq, params)

    preprocessed_data_path = get_preprocessed_data_path(filename, params)
    save_features(features, duration, preprocessed_data_path,
                  data_format=cache_format)

  return features, duration


def normalize_signal(signal):
  """
  Normalize float32 signal to [-1, 1] range
  """
  return signal / (np.max(np.abs(signal)) + 1e-5)


def augment_audio_signal(signal, sample_freq, augmentation):
  """Function that performs audio signal augmentation.

  Args:
    signal (np.array): np.array containing raw audio signal.
    sample_freq (float): frames per second.
    augmentation (dict, optional): None or dictionary of augmentation parameters.
        If not None, has to have 'speed_perturbation_ratio',
        'noise_level_min', or 'noise_level_max' fields, e.g.::
          augmentation={
            'speed_perturbation_ratio': 0.2,
            'noise_level_min': -90,
            'noise_level_max': -46,
          }
        'speed_perturbation_ratio' can either be a list of possible speed
        perturbation factors or a float. If float, a random value from 
        U[1-speed_perturbation_ratio, 1+speed_perturbation_ratio].
  Returns:
    np.array: np.array with augmented audio signal.
  """
  signal_float = normalize_signal(signal.astype(np.float32))

  if 'speed_perturbation_ratio' in augmentation:
    stretch_amount = -1
    if isinstance(augmentation['speed_perturbation_ratio'], list):
      stretch_amount = np.random.choice(augmentation['speed_perturbation_ratio'])
    elif augmentation['speed_perturbation_ratio'] > 0:
      # time stretch (might be slow)
      stretch_amount = 1.0 + (2.0 * np.random.rand() - 1.0) * \
                       augmentation['speed_perturbation_ratio']
    if stretch_amount > 0:
      signal_float = rs.resample(
          signal_float,
          sample_freq,
          int(sample_freq * stretch_amount),
          filter='kaiser_best',
      )

  # noise
  if 'noise_level_min' in augmentation and 'noise_level_max' in augmentation:
    noise_level_db = np.random.randint(low=augmentation['noise_level_min'],
                                       high=augmentation['noise_level_max'])
    signal_float += np.random.randn(signal_float.shape[0]) * \
                    10.0 ** (noise_level_db / 20.0)

  return normalize_signal(signal_float)


def preemphasis(signal, coeff=0.97):
  return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def get_speech_features(signal, sample_freq, params):
  """
  Get speech features using either librosa (recommended) or
  python_speech_features
  Args:
    signal (np.array): np.array containing raw audio signal
    sample_freq (float): sample rate of the signal
    params (dict): parameters of pre-processing
  Returns:
    np.array: np.array of audio features with shape=[num_time_steps,
    num_features].
    audio_duration (float): duration of the signal in seconds
  """

  backend = params.get('backend', 'psf')

  features_type = params.get('input_type', 'spectrogram')
  num_features = params['num_audio_features']
  window_size = params.get('window_size', 20e-3)
  window_stride = params.get('window_stride', 10e-3)
  augmentation = params.get('augmentation', None)

  if backend == 'librosa':
    window_fn = WINDOWS_FNS[params.get('window', "hanning")]
    dither = params.get('dither', 0.0)
    num_fft = params.get('num_fft', None)
    norm_per_feature = params.get('norm_per_feature', False)
    mel_basis = params.get('mel_basis', None)
    if mel_basis is not None and sample_freq != params["sample_freq"]:
      raise ValueError(
          ("The sampling frequency set in params {} does not match the "
           "frequency {} read from file {}").format(params["sample_freq"],
                                                    sample_freq, filename)
      )
    features, duration = get_speech_features_librosa(
        signal, sample_freq, num_features, features_type,
        window_size, window_stride, augmentation, window_fn=window_fn,
        dither=dither, norm_per_feature=norm_per_feature, num_fft=num_fft,
        mel_basis=mel_basis
    )
  else:
    pad_to = params.get('pad_to', 8)
    features, duration = get_speech_features_psf(
        signal, sample_freq, num_features, pad_to, features_type,
        window_size, window_stride, augmentation
    )

  return features, duration 


def get_speech_features_librosa(signal, sample_freq, num_features,
                                features_type='spectrogram',
                                window_size=20e-3,
                                window_stride=10e-3,
                                augmentation=None,
                                window_fn=np.hanning,
                                num_fft=None,
                                dither=0.0,
                                norm_per_feature=False,
                                mel_basis=None):
  """Function to convert raw audio signal to numpy array of features.
  Backend: librosa
  Args:
    signal (np.array): np.array containing raw audio signal.
    sample_freq (float): frames per second.
    num_features (int): number of speech features in frequency domain.
    pad_to (int): if specified, the length will be padded to become divisible
        by ``pad_to`` parameter.
    features_type (string): 'mfcc' or 'spectrogram'.
    window_size (float): size of analysis window in milli-seconds.
    window_stride (float): stride of analysis window in milli-seconds.
    augmentation (dict, optional): dictionary of augmentation parameters. See
        :func:`augment_audio_signal` for specification and example.

  Returns:
    np.array: np.array of audio features with shape=[num_time_steps,
    num_features].
    audio_duration (float): duration of the signal in seconds
  """
  if augmentation:
    signal = augment_audio_signal(signal.astype(np.float32), sample_freq, augmentation)
  else:
    signal = normalize_signal(signal.astype(np.float32))

  audio_duration = len(signal) * 1.0 / sample_freq

  n_window_size = int(sample_freq * window_size)
  n_window_stride = int(sample_freq * window_stride)
  num_fft = num_fft or 2**math.ceil(math.log2(window_size*sample_freq))

  if dither > 0:
    signal += dither*np.random.randn(*signal.shape)

  if features_type == 'spectrogram':
    # ignore 1/n_fft multiplier, since there is a post-normalization
    powspec = np.square(np.abs(librosa.core.stft(
        signal, n_fft=n_window_size,
        hop_length=n_window_stride, win_length=n_window_size, center=True,
        window=window_fn)))
    # remove small bins
    powspec[powspec <= 1e-30] = 1e-30
    features = 10 * np.log10(powspec.T)

    assert num_features <= n_window_size // 2 + 1, \
      "num_features for spectrogram should be <= (sample_freq * window_size // 2 + 1)"

    # cut high frequency part
    features = features[:, :num_features]

  elif features_type == 'mfcc':
    signal = preemphasis(signal, coeff=0.97)
    S = np.square(
            np.abs(
                librosa.core.stft(signal, n_fft=num_fft,
                                  hop_length=int(window_stride * sample_freq),
                                  win_length=int(window_size * sample_freq),
                                  center=True, window=window_fn
                )
            )
        )
    features = librosa.feature.mfcc(sr=sample_freq, S=S,
        n_mfcc=num_features, n_mels=2*num_features).T
  elif features_type == 'logfbank':
    signal = preemphasis(signal,coeff=0.97)
    S = np.abs(librosa.core.stft(signal, n_fft=num_fft,
                                 hop_length=int(window_stride * sample_freq),
                                 win_length=int(window_size * sample_freq),
                                 center=True, window=window_fn))**2.0
    if mel_basis is None:
      # Build a Mel filter
      mel_basis = librosa.filters.mel(sample_freq, num_fft, n_mels=num_features,
                                      fmin=0, fmax=int(sample_freq/2))
    features = np.log(np.dot(mel_basis, S) + 1e-20).T

  else:
    raise ValueError('Unknown features type: {}'.format(features_type))

  norm_axis = 0 if norm_per_feature else None
  mean = np.mean(features, axis=norm_axis)
  std_dev = np.std(features, axis=norm_axis)
  features = (features - mean) / std_dev

  if augmentation:
    n_freq_mask = augmentation.get('n_freq_mask', 0)
    n_time_mask = augmentation.get('n_time_mask', 0)
    width_freq_mask = augmentation.get('width_freq_mask', 10)
    width_time_mask = augmentation.get('width_time_mask', 50)

    for idx in range(n_freq_mask):
      freq_band = np.random.randint(width_freq_mask + 1)
      freq_base = np.random.randint(0, features.shape[1] - freq_band)
      features[:, freq_base:freq_base+freq_band] = 0
    for idx in range(n_time_mask):
      time_band = np.random.randint(width_time_mask + 1)
      if features.shape[0] - time_band > 0:
        time_base = np.random.randint(features.shape[0] - time_band)
        features[time_base:time_base+time_band, :] = 0

  # now it is safe to pad
  # if pad_to > 0:
  #   if features.shape[0] % pad_to != 0:
  #     pad_size = pad_to - features.shape[0] % pad_to
  #     if pad_size != 0:
  #         features = np.pad(features, ((0,pad_size), (0,0)), mode='constant')
  return features, audio_duration


def get_speech_features_psf(signal, sample_freq, num_features,
                            pad_to=8,
                            features_type='spectrogram',
                            window_size=20e-3,
                            window_stride=10e-3,
                            augmentation=None):
  """Function to convert raw audio signal to numpy array of features.
  Backend: python_speech_features
  Args:
    signal (np.array): np.array containing raw audio signal.
    sample_freq (float): frames per second.
    num_features (int): number of speech features in frequency domain.
    pad_to (int): if specified, the length will be padded to become divisible
        by ``pad_to`` parameter.
    features_type (string): 'mfcc' or 'spectrogram'.
    window_size (float): size of analysis window in milli-seconds.
    window_stride (float): stride of analysis window in milli-seconds.
    augmentation (dict, optional): dictionary of augmentation parameters. See
        :func:`augment_audio_signal` for specification and example.
    apply_window (bool): whether to apply Hann window for mfcc and logfbank.
        python_speech_features version should accept winfunc if it is True.
  Returns:
    np.array: np.array of audio features with shape=[num_time_steps,
    num_features].
    audio_duration (float): duration of the signal in seconds
  """
  if augmentation is not None:
    signal = augment_audio_signal(signal, sample_freq, augmentation)
  else:
    signal = (normalize_signal(signal.astype(np.float32)) * 32767.0).astype(
        np.int16)

  audio_duration = len(signal) * 1.0 / sample_freq

  n_window_size = int(sample_freq * window_size)
  n_window_stride = int(sample_freq * window_stride)

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
      "num_features for spectrogram should be <= (sample_freq * window_size // 2 + 1)"

    # cut high frequency part
    features = features[:, :num_features]

  elif features_type == 'mfcc':
    features = psf.mfcc(signal=signal,
                        samplerate=sample_freq,
                        winlen=window_size,
                        winstep=window_stride,
                        numcep=num_features,
                        nfilt=2 * num_features,
                        nfft=512,
                        lowfreq=0, highfreq=None,
                        preemph=0.97,
                        ceplifter=2 * num_features,
                        appendEnergy=False)

  elif features_type == 'logfbank':
    features = psf.logfbank(signal=signal,
                            samplerate=sample_freq,
                            winlen=window_size,
                            winstep=window_stride,
                            nfilt=num_features,
                            nfft=512,
                            lowfreq=0, highfreq=sample_freq / 2,
                            preemph=0.97)
  else:
    raise ValueError('Unknown features type: {}'.format(features_type))

  if pad_to > 0:
    assert features.shape[0] % pad_to == 0
  mean = np.mean(features)
  std_dev = np.std(features)
  features = (features - mean) / std_dev

  return features, audio_duration

