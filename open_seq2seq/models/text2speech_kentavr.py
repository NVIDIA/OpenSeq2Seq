# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import matplotlib as mpl
import numpy as np
from six.moves import range

mpl.use("Agg")

from .encoder_decoder import EncoderDecoderModel
from .text2speech import plot_spectrograms, save_audio


class Text2SpeechKentavr(EncoderDecoderModel):
  """
  Text-to-speech data layer class for Kentavr model.
  """

  def __init__(self, params, mode="train", hvd=None):
    super(Text2SpeechKentavr, self).__init__(params, mode=mode, hvd=hvd)
    self._save_to_tensorboard = self.params["save_to_tensorboard"]

  def maybe_print_logs(self, input_values, output_values, training_step):
    dict_to_log = {}
    step = training_step
    spec, stop_target, _ = input_values["target_tensors"]
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

    alignment_specs, alignment_titles = self._get_alignments(alignment)
    specs += alignment_specs
    titles += alignment_titles

    if "both" in self.get_data_layer().params["output_type"]:
      specs.append(output_values[5][0])
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

    im_summary = plot_spectrograms(
      specs,
      titles,
      stop_token_pred,
      audio_length,
      self.params["logdir"],
      step,
      append="train",
      save_to_tensorboard=self._save_to_tensorboard,
      stop_token_target=stop_target
    )

    dict_to_log["image"] = im_summary

    if self._save_to_tensorboard:
      save_format = "tensorboard"
    else:
      save_format = "disk"
    if "both" in self.get_data_layer().params["output_type"]:
      predicted_mag_spec = output_values[5][0][:audio_length - 1, :]
      if self.get_data_layer()._exp_mag is False:
        predicted_mag_spec = np.exp(predicted_mag_spec)
      predicted_mag_spec = self.get_data_layer().get_magnitude_spec(predicted_mag_spec)
      wav_summary = save_audio(
        predicted_mag_spec,
        self.params["logdir"],
        step,
        n_fft=self.get_data_layer().n_fft,
        sampling_rate=self.get_data_layer().sampling_rate,
        mode="train_mag",
        save_format=save_format,
        max_normalization=self.get_data_layer().max_normalization
      )
      dict_to_log["audio_mag"] = wav_summary
    predicted_final_spec = predicted_final_spec[:audio_length - 1, :]
    predicted_final_spec = self.get_data_layer().get_magnitude_spec(predicted_final_spec, is_mel=True)
    wav_summary = save_audio(
      predicted_final_spec,
      self.params["logdir"],
      step,
      n_fft=self.get_data_layer().n_fft,
      sampling_rate=self.get_data_layer().sampling_rate,
      save_format=save_format,
      max_normalization=self.get_data_layer().max_normalization
    )
    dict_to_log["audio"] = wav_summary

    if self._save_to_tensorboard:
      return dict_to_log
    return {}

  def finalize_evaluation(self, results_per_batch, training_step=None, samples_count=1):
    dict_to_log = {}
    step = training_step
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

    alignment_specs, alignment_titles = self._get_alignments(attention_mask)
    specs += alignment_specs
    titles += alignment_titles

    if "both" in self.get_data_layer().params["output_type"]:
      n_feats = self.get_data_layer().params["num_audio_features"]
      mag_pred = output_values[5]
      mag_pred_pad = np.zeros(
        [max_length - np.shape(mag_pred)[0], n_feats["magnitude"]]
      )
      mag_pred = np.concatenate([mag_pred, mag_pred_pad], axis=0)
      specs.append(mag_pred)
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

    im_summary = plot_spectrograms(
      specs,
      titles,
      stop_token_pred,
      audio_length,
      self.params["logdir"],
      step,
      append="eval",
      save_to_tensorboard=self._save_to_tensorboard,
      stop_token_target=stop_target
    )

    dict_to_log["image"] = im_summary

    if audio_length > 2:
      if self._save_to_tensorboard:
        save_format = "tensorboard"
      else:
        save_format = "disk"
      if "both" in self.get_data_layer().params["output_type"]:
        predicted_mag_spec = output_values[5][:audio_length - 1, :]
        if self.get_data_layer()._exp_mag is False:
          predicted_mag_spec = np.exp(predicted_mag_spec)
        predicted_mag_spec = self.get_data_layer().get_magnitude_spec(predicted_mag_spec)
        wav_summary = save_audio(
          predicted_mag_spec,
          self.params["logdir"],
          step,
          n_fft=self.get_data_layer().n_fft,
          sampling_rate=self.get_data_layer().sampling_rate,
          mode="eval_mag",
          save_format=save_format,
          max_normalization=self.get_data_layer().max_normalization
        )
        dict_to_log["audio_mag"] = wav_summary
      predicted_final_spec = predicted_final_spec[:audio_length - 1, :]
      predicted_final_spec = self.get_data_layer().get_magnitude_spec(predicted_final_spec, is_mel=True)
      wav_summary = save_audio(
        predicted_final_spec,
        self.params["logdir"],
        step,
        n_fft=self.get_data_layer().n_fft,
        sampling_rate=self.get_data_layer().sampling_rate,
        mode="eval",
        save_format=save_format,
        max_normalization=self.get_data_layer().max_normalization
      )
      dict_to_log["audio"] = wav_summary

      return dict_to_log

    return {}

  def evaluate(self, input_values, output_values):
    # Need to reduce amount of data sent for horovod
    # Use last element
    idx = -1
    output_values = [(item[idx]) for item in output_values]
    input_values = {
      key: [value[0][idx], value[1][idx]] for key, value in input_values.items()
    }
    return [input_values, output_values]

  def infer(self, input_values, output_values):
    if self.on_horovod:
      raise ValueError("Inference is not supported on horovod")

    return [input_values, output_values]

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

        alignment_specs, alignment_titles = self._get_alignments(attention_mask_sample)
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

  @staticmethod
  def _get_alignments(attention_mask):
    alignments_name = ["dec_enc_alignment"]

    specs = []
    titles = []

    for name, alignment in zip(alignments_name, attention_mask):
      for layer in range(len(alignment)):
        for head in range(alignment.shape[1]):
          specs.append(alignment[layer][head])
          titles.append("{}_layer_{}_head_{}".format(name, layer, head))

    return specs, titles

  @staticmethod
  def get_required_params():
    return dict(
      EncoderDecoderModel.get_required_params(), **{
        "save_to_tensorboard": bool,
      }
    )
