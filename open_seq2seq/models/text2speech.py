# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import pandas as pd
import tensorflow as tf
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io.wavfile import write
import librosa

from .encoder_decoder import EncoderDecoderModel
from open_seq2seq.utils.utils import deco_print


def plot_spectrogram(ground_truth, generated_sample, post_net_sample, attention,
 logdir, train_step, number=0, append=False, vmin=None, vmax=None):
  fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(8,12))
  
  if vmin is None:
    vmin = min(np.min(ground_truth), np.min(generated_sample), np.min(post_net_sample))
  if vmax is None:
    vmax = max(np.max(ground_truth), np.max(generated_sample), np.min(post_net_sample))
  
  colour1 = ax1.imshow(ground_truth.T, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
  colour2 = ax2.imshow(generated_sample.T, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
  colour3 = ax3.imshow(post_net_sample.T, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
  colour4 = ax4.imshow(attention.T, cmap='viridis', interpolation='nearest', aspect='auto')
  
  ax1.invert_yaxis()
  ax1.set_ylabel('fourier components')
  ax1.set_title('training data')
  
  ax2.invert_yaxis()
  ax2.set_ylabel('fourier components')
  ax2.set_title('decoder results')

  ax3.invert_yaxis()
  ax3.set_ylabel('fourier components')
  ax3.set_title('post net results')

  ax4.set_title('attention')
  ax4.set_ylabel('inputs')
  
  plt.xlabel('time')
  
  fig.subplots_adjust(right=0.8)
  cbar_ax1 = fig.add_axes([0.85, 0.35, 0.05, 0.5])
  fig.colorbar(colour1, cax=cbar_ax1)
  cbar_ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.15])
  fig.colorbar(colour4, cax=cbar_ax2)

  if append:
    name = '{}/Output_Step{}_{}_{}.png'.format(logdir, train_step, number, append)
  else:
    name = '{}/Output_Step{}_{}.png'.format(logdir, train_step, number)
  if logdir[0] != '/':
    name = "./"+name
  #save
  fig.savefig(name, dpi=300)

  plt.close(fig)

def save_audio(mag_spec, logdir, name, train=True):
  magnitudes = np.exp(mag_spec)
  signal = griffin_lim(magnitudes.T**1.2)
  if train:
    file_name = '{}/sample_train_step_{}.wav'.format(logdir, name)
  else:
    file_name = '{}/sample_eval_step_{}.wav'.format(logdir, name)
  if logdir[0] != '/':
    file_name = "./"+file_name
  write(file_name, 22050 ,signal)

def griffin_lim(magnitudes, n_iters=50):
  """
  PARAMS
  ------
  magnitudes: spectrogram magnitudes
  stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
  """

  phase = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape))
  complex_spec = magnitudes * phase
  signal = librosa.istft(complex_spec)

  for i in range(n_iters):
    _, phase = librosa.magphase(librosa.stft(signal, n_fft=1024))
    complex_spec = magnitudes * phase
    signal = librosa.istft(complex_spec)
  return signal

def sparse_tensor_to_chars(tensor, idx2char):
  text = [''] * tensor.dense_shape[0]
  for idx_tuple, value in zip(tensor.indices, tensor.values):
    text[idx_tuple[0]] += idx2char[value]
  return text

class Text2Speech(EncoderDecoderModel):
  def _create_decoder(self):
    self.params['decoder_params']['num_audio_features'] = (
      self.get_data_layer().params['num_audio_features']
    )
    return super(Text2Speech, self)._create_decoder()

  def maybe_print_logs(self, input_values, output_values, step):
    y, y_length = input_values['target_tensors']
    predicted_decoder_spectrograms = output_values[0]
    predicted_final_spectrograms = output_values[1]
    attention_mask = output_values[2]
    y_sample = y[0]
    y_length_sample = y_length[0]
    predicted_spectrogram_sample = predicted_decoder_spectrograms[0]
    predicted_final_spectrogram_sample = predicted_final_spectrograms[0]
    attention_mask_sample = attention_mask[0]

    plot_spectrogram(y_sample, predicted_spectrogram_sample,
                     predicted_final_spectrogram_sample,
                     attention_mask_sample,
                     self.params["logdir"], step,
                     append="train")

    if self.get_data_layer().params['output_type'] == "spectrogram":
      save_audio(predicted_final_spectrogram_sample, self.params["logdir"], step)
    
    return {}

  def _build_forward_pass_graph(self, input_tensors, gpu_id=0):
    """TensorFlow graph for sequence-to-sequence model is created here.
    This function connects encoder, decoder and loss together. As an input for
    encoder it will specify source sequence and source length (as returned from
    the data layer). As an input for decoder it will specify target sequence
    and target length as well as all output returned from encoder. For loss it
    will also specify target sequence and length and all output returned from
    decoder. Note that loss will only be built for mode == "train" or "eval".

    See :meth:`models.model.Model._build_forward_pass_graph` for description of
    arguments and return values.
    """
    if not isinstance(input_tensors, dict) or \
       'source_tensors' not in input_tensors:
      raise ValueError('Input tensors should be a dict containing '
                       '"source_tensors" key')

    if not isinstance(input_tensors['source_tensors'], list):
      raise ValueError('source_tensors should be a list')

    source_tensors = input_tensors['source_tensors']
    if self.mode == "train" or self.mode == "eval":
      if 'target_tensors' not in input_tensors:
        raise ValueError('Input tensors should contain "target_tensors" key'
                         'when mode != "infer"')
      if not isinstance(input_tensors['target_tensors'], list):
        raise ValueError('target_tensors should be a list')
      target_tensors = input_tensors['target_tensors']

    with tf.variable_scope("ForwardPass"):
      encoder_input = {"source_tensors": source_tensors}
      encoder_output = self.encoder.encode(input_dict=encoder_input)

      decoder_input = {"encoder_output": encoder_output}
      # if self.mode == "train":
      decoder_input['target_tensors'] = target_tensors
      decoder_output = self.decoder.decode(input_dict=decoder_input)
      decoder_out = decoder_output.get("decoder_output", 0)
      decoder_post_net_outupt = decoder_output.get("post_net_output", 0)
      attention_mask = decoder_output.get("alignments", 0)

      final_spectrogram = decoder_out + decoder_post_net_outupt

      if self.mode == "train" or self.mode == "eval":
        with tf.variable_scope("Loss"):
          loss_input_dict = {
            "decoder_output": decoder_output,
            "target_tensors": target_tensors,
          }
          loss = self.loss_computator.compute_loss(loss_input_dict)
      else:
        deco_print("Inference Mode. Loss part of graph isn't built.")
        loss = None
      return loss, [decoder_out, final_spectrogram, attention_mask]

  def finalize_evaluation(self, results_per_batch, step):
    sample = results_per_batch[-1]
    input_values = sample[0]
    output_values = sample[1]
    y, y_length = input_values['target_tensors']
    predicted_decoder_spectrograms = output_values[0]
    predicted_final_spectrograms = output_values[1]
    attention_mask = output_values[2]

    y_sample = y[-1]
    predicted_spectrogram_sample = predicted_decoder_spectrograms[-1]
    predicted_final_spectrogram_sample = predicted_final_spectrograms[-1]
    attention_mask_sample = attention_mask[-1]

    plot_spectrogram(y_sample, predicted_spectrogram_sample,
                     predicted_final_spectrogram_sample,
                     attention_mask_sample,
                     self.params["logdir"], step,
                     append="eval")

    if self.get_data_layer().params['output_type'] == "spectrogram":
      save_audio(predicted_final_spectrogram_sample, self.params["logdir"], step, train=False)

    return {}


  def evaluate(self, input_values, output_values):
    return [input_values, output_values]

  # def infer(self, input_values, output_values):
  #   preds = []
  #   decoded_sequence = output_values[0]
  #   decoded_texts = sparse_tensor_to_chars(
  #     decoded_sequence,
  #     self.get_data_layer().params['idx2char'],
  #   )
  #   for sample_id in range(len(decoded_texts)):
  #     preds.append("".join(decoded_texts[sample_id]))
  #   return preds

  # def finalize_inference(self, results_per_batch, output_file):
  #   preds = []

  #   for result in results_per_batch:
  #     preds.extend(result)
  #   pd.DataFrame(
  #     {
  #       'wav_filename': self.get_data_layer().all_files,
  #       'predicted_transcript': preds,
  #     },
  #     columns=['wav_filename', 'predicted_transcript'],
  #   ).to_csv(output_file, index=False)
