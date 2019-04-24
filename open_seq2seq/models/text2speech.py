# Copyright (c) 2019 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import librosa
import matplotlib as mpl
import numpy as np
from scipy.io.wavfile import write
from six import BytesIO
from six.moves import range

mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from .encoder_decoder import EncoderDecoderModel


def plot_spectrograms(
        specs,
        titles,
        stop_token_pred,
        audio_length,
        logdir,
        train_step,
        stop_token_target=None,
        number=0,
        append=False,
        save_to_tensorboard=False
):
  """
  Helper function to create a image to be logged to disk or a tf.Summary to be
  logged to tensorboard.

  Args:
    specs (array): array of images to show
    titles (array): array of titles. Must match lengths of specs array
    stop_token_pred (np.array): np.array of size [time, 1] containing the stop
      token predictions from the model.
    audio_length (int): lenth of the predicted spectrogram
    logdir (str): dir to save image file is save_to_tensorboard is disabled.
    train_step (int): current training step
    stop_token_target (np.array): np.array of size [time, 1] containing the stop
      token target.
    number (int): Current sample number (used if evaluating more than 1 sample
      from a batch)
    append (str): Optional string to append to file name eg. train, eval, infer
    save_to_tensorboard (bool): If False, the created image is saved to the
      logdir as a png file. If True, the function returns a tf.Summary object
      containing the image and will be logged to the current tensorboard file.

  Returns:
    tf.Summary or None
  """
  num_figs = len(specs) + 1
  fig, ax = plt.subplots(nrows=num_figs, figsize=(8, num_figs * 3))

  for i, (spec, title) in enumerate(zip(specs, titles)):
    spec = np.pad(spec, ((1, 1), (1, 1)), "constant", constant_values=0.)
    spec = spec.astype(float)
    colour = ax[i].imshow(
      spec.T, cmap='viridis', interpolation=None, aspect='auto'
    )
    ax[i].invert_yaxis()
    ax[i].set_title(title)
    fig.colorbar(colour, ax=ax[i])
  if stop_token_target is not None:
    stop_token_target = stop_token_target.astype(float)
    ax[-1].plot(stop_token_target, 'r.')
  stop_token_pred = stop_token_pred.astype(float)
  ax[-1].plot(stop_token_pred, 'g.')
  ax[-1].axvline(x=audio_length)
  ax[-1].set_xlim(0, len(specs[0]))
  ax[-1].set_title("stop token")

  plt.xlabel('time')
  plt.tight_layout()

  cb = fig.colorbar(colour, ax=ax[-1])
  cb.remove()

  if save_to_tensorboard:
    tag = "{}_image".format(append)
    iostream = BytesIO()
    fig.savefig(iostream, dpi=300)
    summary = tf.Summary.Image(
        encoded_image_string=iostream.getvalue(),
        height=int(fig.get_figheight() * 300),
        width=int(fig.get_figwidth() * 300)
    )
    summary = tf.Summary.Value(tag=tag, image=summary)
    plt.close(fig)

    return summary
  else:
    if append:
      name = '{}/Output_step{}_{}_{}.png'.format(
          logdir, train_step, number, append
      )
    else:
      name = '{}/Output_step{}_{}.png'.format(logdir, train_step, number)
    if logdir[0] != '/':
      name = "./" + name
    # save
    fig.savefig(name, dpi=300)

    plt.close(fig)
    return None


def save_audio(
        magnitudes,
        logdir,
        step,
        sampling_rate,
        n_fft=1024,
        mode="train",
        number=0,
        save_format="tensorboard",
        power=1.5,
        gl_iters=50,
        verbose=True,
        max_normalization=False
):
  """
  Helper function to create a wav file to be logged to disk or a tf.Summary to
  be logged to tensorboard.

  Args:
    magnitudes (np.array): np.array of size [time, n_fft/2 + 1] containing the
      energy spectrogram.
    logdir (str): dir to save image file is save_to_tensorboard is disabled.
    step (int): current training step
    n_fft (int): number of filters for fft and ifft.
    sampling_rate (int): samplng rate in Hz of the audio to be saved.
    number (int): Current sample number (used if evaluating more than 1 sample
    mode (str): Optional string to append to file name eg. train, eval, infer
      from a batch)
    save_format: save_audio can either return the np.array containing the
      generated sound, log the wav file to the disk, or return a tensorboard
      summary object. Each method can be enabled by passing save_format as
      "np.array", "tensorboard", or "disk" respectively.

  Returns:
    tf.Summary or None
  """
  # Clip signal max and min
  if np.min(magnitudes) < 0 or np.max(magnitudes) > 255:
    if verbose:
      print("WARNING: {} audio was clipped at step {}".format(mode.capitalize(), step))
    magnitudes = np.clip(magnitudes, a_min=0, a_max=255)
  signal = griffin_lim(magnitudes.T ** power, n_iters=gl_iters, n_fft=n_fft)

  if max_normalization:
    signal /= np.max(np.abs(signal))

  if save_format == "np.array":
    return signal
  elif save_format == "tensorboard":
    tag = "{}_audio".format(mode)
    iostream = BytesIO()
    write(iostream, sampling_rate, signal)
    summary = tf.Summary.Audio(encoded_audio_string=iostream.getvalue())
    summary = tf.Summary.Value(tag=tag, audio=summary)
    return summary
  elif save_format == "disk":
    file_name = '{}/sample_step{}_{}_{}.wav'.format(logdir, step, number, mode)
    if logdir[0] != '/':
      file_name = "./" + file_name
    write(file_name, sampling_rate, signal)
    return None
  else:
    print((
            "WARN: The save format passed to save_audio was not understood. No "
            "sound files will be saved for the current step. "
            "Received '{}'."
            "Expected one of 'np.array', 'tensorboard', or 'disk'"
          ).format(save_format))
    return None


def griffin_lim(magnitudes, n_iters=50, n_fft=1024):
  """
  Griffin-Lim algorithm to convert magnitude spectrograms to audio signals
  """

  phase = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape))
  complex_spec = magnitudes * phase
  signal = librosa.istft(complex_spec)
  if not np.isfinite(signal).all():
    print("WARNING: audio was not finite, skipping audio saving")
    return np.array([0])

  for _ in range(n_iters):
    _, phase = librosa.magphase(librosa.stft(signal, n_fft=n_fft))
    complex_spec = magnitudes * phase
    signal = librosa.istft(complex_spec)
  return signal


class Text2Speech(EncoderDecoderModel):
  """
  Text-to-speech data layer.
  """

  @staticmethod
  def get_required_params():
    return dict(
        EncoderDecoderModel.get_required_params(), **{
            "save_to_tensorboard": bool,
        }
    )

  def __init__(self, params, mode="train", hvd=None):
    super(Text2Speech, self).__init__(params, mode=mode, hvd=hvd)
    self._save_to_tensorboard = self.params["save_to_tensorboard"]

  def print_logs(self,
                 mode,
                 specs,
                 titles,
                 stop_token_pred,
                 stop_target,
                 audio_length,
                 step,
                 predicted_final_spec,
                 predicted_mag_spec=None):
    """
    Save audio files and plots.

    Args:
      mode: "train" or "eval".
      specs: spectograms to plot.
      titles: spectogram titles.
      stop_token_pred: stop token prediction.
      stop_target: stop target.
      audio_length: length of the audio.
      step: current step.
      predicted_final_spec: predicted mel spectogram.
      predicted_mag_spec: predicted magnitude spectogram.

    Returns:
      Dictionary to log.
    """

    dict_to_log = {}

    im_summary = plot_spectrograms(
        specs,
        titles,
        stop_token_pred,
        audio_length,
        self.params["logdir"],
        step,
        append=mode,
        save_to_tensorboard=self._save_to_tensorboard,
        stop_token_target=stop_target
    )

    dict_to_log['image'] = im_summary

    if audio_length < 3:
      return {}

    if self._save_to_tensorboard:
      save_format = "tensorboard"
    else:
      save_format = "disk"

    if predicted_mag_spec is not None:
      predicted_mag_spec = predicted_mag_spec[:audio_length - 1, :]

      if self.get_data_layer()._exp_mag is False:
        predicted_mag_spec = np.exp(predicted_mag_spec)

      predicted_mag_spec = self.get_data_layer().get_magnitude_spec(predicted_mag_spec)
      wav_summary = save_audio(
          predicted_mag_spec,
          self.params["logdir"],
          step,
          n_fft=self.get_data_layer().n_fft,
          sampling_rate=self.get_data_layer().sampling_rate,
          mode=mode + "_mag",
          save_format=save_format
      )
      dict_to_log['audio_mag'] = wav_summary

    predicted_final_spec = predicted_final_spec[:audio_length - 1, :]
    predicted_final_spec = self.get_data_layer().get_magnitude_spec(
        predicted_final_spec,
        is_mel=True
    )
    wav_summary = save_audio(
        predicted_final_spec,
        self.params["logdir"],
        step,
        n_fft=self.get_data_layer().n_fft,
        sampling_rate=self.get_data_layer().sampling_rate,
        mode=mode,
        save_format=save_format,
        max_normalization=self.get_data_layer().max_normalization
    )
    dict_to_log['audio'] = wav_summary

    if self._save_to_tensorboard:
      return dict_to_log

    return {}

  def infer(self, input_values, output_values):
    if self.on_horovod:
      raise ValueError("Inference is not supported on horovod")

    return [input_values, output_values]

  def evaluate(self, input_values, output_values):
    # Need to reduce amount of data sent for horovod
    # Use last element
    idx = -1
    output_values = [(item[idx]) for item in output_values]
    input_values = {
        key: [value[0][idx], value[1][idx]] for key, value in input_values.items()
    }
    return [input_values, output_values]

  def get_alignments(self, attention_mask):
    """
    Get attention alignment plots.

    Args:
      attention_mask: attention alignment.

    Returns:
      Specs and titles to plot.
    """

    raise NotImplementedError()

  def finalize_inference(self, results_per_batch, output_file):
    print("output_file is ignored for tts")
    print("results are logged to the logdir")

    batch_size = len(results_per_batch[0][0]["source_tensors"][0])
    for i, sample in enumerate(results_per_batch):
      output_values = sample[1]
      predicted_final_specs = output_values[1]
      attention_mask = output_values[2]
      stop_tokens = output_values[3]
      sequence_lengths = output_values[4]

      for j in range(len(predicted_final_specs)):
        predicted_final_spec = predicted_final_specs[j]
        attention_mask_sample = attention_mask[j]
        stop_tokens_sample = stop_tokens[j]

        specs = [predicted_final_spec]
        titles = ["final spectrogram"]
        audio_length = sequence_lengths[j]

        alignment_specs, alignment_titles = self.get_alignments(attention_mask_sample)
        specs += alignment_specs
        titles += alignment_titles

        if "mel" in self.get_data_layer().params["output_type"]:
          mag_spec = self.get_data_layer().get_magnitude_spec(predicted_final_spec)
          log_mag_spec = np.log(np.clip(mag_spec, a_min=1e-5, a_max=None))
          specs.append(log_mag_spec)
          titles.append("magnitude spectrogram")
        elif "both" in self.get_data_layer().params["output_type"]:
          mag_spec = self.get_data_layer().get_magnitude_spec(predicted_final_spec, is_mel=True)
          specs.append(mag_spec)
          titles.append("mag spectrogram from mel basis")
          specs.append(output_values[5][j])
          titles.append("mag spectrogram from proj layer")

        im_summary = plot_spectrograms(
            specs,
            titles,
            stop_tokens_sample,
            audio_length,
            self.params["logdir"],
            0,
            number=i * batch_size + j,
            append="infer"
        )

        if audio_length > 2:
          if "both" in self.get_data_layer().params["output_type"]:
            predicted_mag_spec = output_values[5][j][:audio_length - 1, :]
            wav_summary = save_audio(
              predicted_mag_spec,
              self.params["logdir"],
              0,
              n_fft=self.get_data_layer().n_fft,
              sampling_rate=self.get_data_layer().sampling_rate,
              mode="infer_mag",
              number=i * batch_size + j,
              save_format="disk",
              max_normalization=self.get_data_layer().max_normalization
            )
          predicted_final_spec = predicted_final_spec[:audio_length - 1, :]
          predicted_final_spec = self.get_data_layer().get_magnitude_spec(predicted_final_spec, is_mel=True)
          wav_summary = save_audio(
              predicted_final_spec,
              self.params["logdir"],
              0,
              n_fft=self.get_data_layer().n_fft,
              sampling_rate=self.get_data_layer().sampling_rate,
              mode="infer",
              number=i * batch_size + j,
              save_format="disk",
              max_normalization=self.get_data_layer().max_normalization
          )

  def finalize_evaluation(self, results_per_batch, training_step=None, samples_count=1):
    sample = results_per_batch[0]

    input_values = sample[0]
    output_values = sample[1]
    y_sample, stop_target = input_values["target_tensors"]
    predicted_spec = output_values[0]
    predicted_final_spec = output_values[1]
    attention_mask = output_values[2]
    stop_token_pred = output_values[3]
    audio_length = output_values[4]

    max_length = np.max([
      y_sample.shape[0],
      predicted_final_spec.shape[0],
    ])
    predictions_pad = np.zeros(
        [max_length - np.shape(predicted_final_spec)[0], np.shape(predicted_final_spec)[-1]]
    )
    stop_token_pred_pad = np.zeros(
        [max_length - np.shape(predicted_final_spec)[0], 1]
    )
    spec_pad = np.zeros([max_length - np.shape(y_sample)[0], np.shape(y_sample)[-1]])
    stop_token_pad = np.zeros([max_length - np.shape(y_sample)[0]])

    predicted_spec = np.concatenate(
        [predicted_spec, predictions_pad], axis=0
    )
    predicted_final_spec = np.concatenate(
        [predicted_final_spec, predictions_pad], axis=0
    )
    stop_token_pred = np.concatenate(
        [stop_token_pred, stop_token_pred_pad], axis=0
    )
    y_sample = np.concatenate([y_sample, spec_pad], axis=0)
    stop_target = np.concatenate([stop_target, stop_token_pad], axis=0)

    specs = [
        y_sample,
        predicted_spec,
        predicted_final_spec
    ]
    titles = [
        "training data",
        "decoder results",
        "post net results"
    ]

    alignment_specs, alignment_titles = self.get_alignments(attention_mask)
    specs += alignment_specs
    titles += alignment_titles

    predicted_mag_spec = None

    if "both" in self.get_data_layer().params["output_type"]:
      n_feats = self.get_data_layer().params["num_audio_features"]
      predicted_mag_spec = output_values[5]
      mag_pred_pad = np.zeros(
          [max_length - np.shape(predicted_mag_spec)[0], n_feats["magnitude"]]
      )
      predicted_mag_spec = np.concatenate([predicted_mag_spec, mag_pred_pad], axis=0)
      specs.append(predicted_mag_spec)
      titles.append("magnitude spectrogram")
      mel, mag = np.split(
          y_sample,
          [n_feats["mel"]],
          axis=1
      )
      specs.insert(0, mel)
      specs[1] = mag
      titles.insert(0, "target mel")
      titles[1] = "target mag"

    return self.print_logs(
        mode="eval",
        specs=specs,
        titles=titles,
        stop_token_pred=stop_token_pred,
        stop_target=stop_target[0],
        audio_length=audio_length,
        step=training_step,
        predicted_final_spec=predicted_final_spec,
        predicted_mag_spec=predicted_mag_spec
    )

  def maybe_print_logs(self, input_values, output_values, training_step):
    spec, stop_target, _ = input_values['target_tensors']
    predicted_decoder_spec = output_values[0]
    predicted_final_spec = output_values[1]
    attention_mask = output_values[2]
    stop_token_pred = output_values[3]
    y_sample = spec[0]
    stop_target = stop_target[0]
    predicted_spec = predicted_decoder_spec[0]
    predicted_final_spec = predicted_final_spec[0]
    alignment = attention_mask[0]
    stop_token_pred = stop_token_pred[0]
    audio_length = output_values[4][0]

    specs = [
        y_sample,
        predicted_spec,
        predicted_final_spec
    ]

    titles = [
        "training data",
        "decoder results",
        "post net results"
    ]

    alignment_specs, alignment_titles = self.get_alignments(alignment)
    specs += alignment_specs
    titles += alignment_titles

    predicted_mag_spec = None

    if "both" in self.get_data_layer().params["output_type"]:
      predicted_mag_spec = output_values[5][0]
      specs.append(predicted_mag_spec)
      titles.append("magnitude spectrogram")
      n_feats = self.get_data_layer().params["num_audio_features"]
      mel, mag = np.split(
          y_sample,
          [n_feats["mel"]],
          axis=1
      )
      specs.insert(0, mel)
      specs[1] = mag
      titles.insert(0, "target mel")
      titles[1] = "target mag"

    return self.print_logs(
        mode="train",
        specs=specs,
        titles=titles,
        stop_token_pred=stop_token_pred,
        stop_target=stop_target,
        audio_length=audio_length,
        step=training_step,
        predicted_final_spec=predicted_final_spec,
        predicted_mag_spec=predicted_mag_spec
    )

