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
from open_seq2seq.data.text2speech.speech_utils import inverse_mel


def plot_spectrogram_w_target(ground_truth, generated_sample, post_net_sample, attention, target_sample, target,
  audio_length, logdir, train_step, number=0, append=False, vmin=None, vmax=None):
  fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, figsize=(8,15))
  
  if vmin is None:
    vmin = min(np.min(ground_truth), np.min(generated_sample), np.min(post_net_sample))
  if vmax is None:
    vmax = max(np.max(ground_truth), np.max(generated_sample), np.max(post_net_sample))
  
  colour1 = ax1.imshow(ground_truth.T, cmap='viridis', interpolation=None, aspect='auto', vmin=vmin, vmax=vmax)
  colour2 = ax2.imshow(generated_sample.T, cmap='viridis', interpolation=None, aspect='auto', vmin=vmin, vmax=vmax)
  colour3 = ax3.imshow(post_net_sample.T, cmap='viridis', interpolation=None, aspect='auto', vmin=vmin, vmax=vmax)
  colour4 = ax4.plot(target_sample, 'g.')
  colour4 = ax4.plot(target, 'r.')
  ax4.axvline(x=audio_length)
  colour5 = ax5.imshow(attention.T, cmap='viridis', interpolation=None, aspect='auto')
  
  ax1.invert_yaxis()
  ax1.set_ylabel('fourier components')
  ax1.set_title('training data')
  
  ax2.invert_yaxis()
  ax2.set_ylabel('fourier components')
  ax2.set_title('decoder results')

  ax3.invert_yaxis()
  ax3.set_ylabel('fourier components')
  ax3.set_title('post net results')

  ax4.set_title('Stop Token Prediction')

  ax5.invert_yaxis()
  ax5.set_title('attention')
  ax5.set_ylabel('inputs')
  
  plt.xlabel('time')

  ax1.axis('off')
  ax2.axis('off')
  ax3.axis('off')
  ax4.axis('off')
  ax5.axis('off')
  
  fig.subplots_adjust(right=0.8)
  cbar_ax1 = fig.add_axes([0.85, 0.45, 0.05, 0.45])
  fig.colorbar(colour1, cax=cbar_ax1)
  cbar_ax3 = fig.add_axes([0.85, 0.1, 0.05, 0.14])
  fig.colorbar(colour5, cax=cbar_ax3)

  if append:
    name = '{}/Output_step{}_{}_{}.png'.format(logdir, train_step, number, append)
  else:
    name = '{}/Output_step{}_{}.png'.format(logdir, train_step, number)
  if logdir[0] != '/':
    name = "./"+name
  #save
  fig.savefig(name, dpi=300)

  plt.close(fig)

def plot_spectrograms(specs, titles, target_sample, audio_length,
  logdir, train_step, number=0, append=False, vmin=None, vmax=None):
  num_figs = len(specs) + 1
  fig, ax = plt.subplots(nrows=num_figs, figsize=(8,num_figs*3))
  
  figures = []
  for i, (spec, title) in enumerate(zip(specs,titles)):
    colour = ax[i].imshow(spec.T, cmap='viridis', interpolation=None, aspect='auto')
    figures.append(colour)
    ax[i].invert_yaxis()
    ax[i].set_title(title)
    ax[i].axis('off')
    fig.colorbar(colour, ax=ax[i])
  target_fig = ax[-1].plot(target_sample, 'g.')
  ax[-1].axvline(x=audio_length)
  
  plt.xlabel('time')
   
  if append:
    name = '{}/Output_step{}_{}_{}.png'.format(logdir, train_step, number, append)
  else:
    name = '{}/Output_step{}_{}.png'.format(logdir, train_step, number)
  if logdir[0] != '/':
    name = "./"+name
  #save
  fig.savefig(name, dpi=300)

  plt.close(fig)

def save_audio(magnitudes, logdir, step, mode="train", number=0):
  # magnitudes = np.exp(mag_spec)
  signal = griffin_lim(magnitudes.T**1.2)
  file_name = '{}/sample_step{}_{}_{}.wav'.format(logdir, step, number, mode)
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

class Text2Speech(EncoderDecoderModel):
  def _create_decoder(self):
    self.params['decoder_params']['num_audio_features'] = (
      self.get_data_layer().params['num_audio_features']
    )
    return super(Text2Speech, self)._create_decoder()

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
      if self.mode != "infer":
        decoder_input['target_tensors'] = target_tensors
      decoder_output = self.decoder.decode(input_dict=decoder_input)
      decoder_out = decoder_output.get("decoder_output", 0.)
      decoder_post_net_outupt = decoder_output.get("post_net_output", 0.)
      attention_mask = decoder_output.get("alignments", 0.)
      target_output = decoder_output.get("target_output", 0.)
      final_sequence_lengths = decoder_output.get("final_sequence_lengths", 0.)
      target_output = tf.sigmoid(target_output)

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
      return loss, [decoder_out, final_spectrogram, attention_mask, target_output, final_sequence_lengths]

  def maybe_print_logs(self, input_values, output_values, step):
    spec, target, _ = input_values['target_tensors']
    predicted_decoder_spectrograms = output_values[0]
    predicted_final_spectrograms = output_values[1]
    attention_mask = output_values[2]
    target_output = output_values[3]
    y_sample = spec[0]
    target = target[0]
    # y_length_sample = y_length[0]
    predicted_spectrogram_sample = predicted_decoder_spectrograms[0]
    predicted_final_spectrogram_sample = predicted_final_spectrograms[0]
    attention_mask_sample = attention_mask[0]
    target_output_sample = target_output[0]
    audio_length = output_values[4][0]

    plot_spectrogram_w_target(y_sample, predicted_spectrogram_sample,
                     predicted_final_spectrogram_sample,
                     attention_mask_sample,
                     target_output_sample,
                     target,
                     audio_length,
                     self.params["logdir"], step,
                     append="train")

    predicted_final_spectrogram_sample = predicted_final_spectrogram_sample[:audio_length-1,:]
    if "spectrogram" in self.get_data_layer().params['output_type']:
      predicted_final_spectrogram_sample = np.exp(predicted_final_spectrogram_sample)
      save_audio(predicted_final_spectrogram_sample, self.params["logdir"], step)
    elif "mel" in self.get_data_layer().params['output_type']:
      predicted_final_spectrogram_sample = inverse_mel(predicted_final_spectrogram_sample)
      save_audio(predicted_final_spectrogram_sample, self.params["logdir"], step)
    
    return {}

  def finalize_evaluation(self, results_per_batch, step):
    sample = results_per_batch[-1]
    input_values = sample[0]
    output_values = sample[1]
    # y, y_length = input_values['target_tensors']
    y_sample, target = input_values['target_tensors']
    predicted_spectrogram_sample = output_values[0]
    predicted_final_spectrogram_sample = output_values[1]
    attention_mask_sample = output_values[2]
    target_output_sample = output_values[3]
    audio_length = output_values[4]

    plot_spectrogram_w_target(y_sample, predicted_spectrogram_sample,
                     predicted_final_spectrogram_sample,
                     attention_mask_sample,
                     target_output_sample,
                     target,
                     audio_length,
                     self.params["logdir"], step,
                     append="eval")

    predicted_final_spectrogram_sample = predicted_final_spectrogram_sample[:audio_length-1,:]
    if audio_length > 2:
      if "spectrogram" in self.get_data_layer().params['output_type']:
        predicted_final_spectrogram_sample = np.exp(predicted_final_spectrogram_sample)
        save_audio(predicted_final_spectrogram_sample, self.params["logdir"], step, mode="eval")
      elif "mel" in self.get_data_layer().params['output_type']:
        predicted_final_spectrogram_sample = inverse_mel(predicted_final_spectrogram_sample)
        save_audio(predicted_final_spectrogram_sample, self.params["logdir"], step, mode="eval")

    return {}


  def evaluate(self, input_values, output_values):
    # Need to reduce amount of data sent for horovod
    output_values = [item[-3] for item in output_values]
    input_values = {key:[value[0][-3],value[1][-3]] for key, value in input_values.items()}
    return [input_values, output_values]

  def infer(self, input_values, output_values):
    if self.on_horovod:
      raise ValueError('Inference is not supported on horovod') 
    return [input_values, output_values]

  def finalize_inference(self, results_per_batch, output_file):
    print("output_file is ignored for ts2")
    batch_size = len(results_per_batch[0][0]['source_tensors'][0])
    for i, sample in enumerate(results_per_batch):
      input_values = sample[0]
      output_values = sample[1]
      # y, y_length = input_values['target_tensors']
      # predicted_decoder_spectrograms = output_values[0]
      predicted_final_spectrograms = output_values[1]
      attention_mask = output_values[2]
      stop_tokens = output_values[3]
      sequence_lengths = output_values[4]

      for j in range(len(predicted_final_spectrograms)):
        # y_sample = y[j]
        # predicted_spectrogram_sample = predicted_decoder_spectrograms[j]
        predicted_final_spectrogram_sample = predicted_final_spectrograms[j]
        attention_mask_sample = attention_mask[j]
        stop_tokens_sample = stop_tokens[j]

        specs = [predicted_final_spectrogram_sample, attention_mask_sample]
        titles = ["final spectrogram", "attention"]
        audio_length = sequence_lengths[j]

        if "mel" in self.get_data_layer().params['output_type']:
          mag_spec = inverse_mel(predicted_final_spectrogram_sample)
          log_mag_spec = np.log(np.clip(mag_spec, a_min=1e-5, a_max=None))
          specs.append(log_mag_spec)
          titles.append("linear spectrogram")

        plot_spectrograms(specs,
                          titles,
                          stop_tokens_sample,
                          audio_length,
                          self.params["logdir"], 0,
                          number= i*batch_size+j,
                          append="infer")

        # print(predicted_final_spectrogram_sample.shape)
        if audio_length > 2:
          if "spectrogram" in self.get_data_layer().params['output_type']:
            predicted_final_spectrogram_sample = predicted_final_spectrogram_sample[:audio_length-1,:]
            predicted_final_spectrogram_sample = np.exp(predicted_final_spectrogram_sample)
            save_audio(predicted_final_spectrogram_sample, self.params["logdir"], 0, mode="infer",number= i*batch_size+j)
          elif "mel" in self.get_data_layer().params['output_type']:
            predicted_final_spectrogram_sample = mag_spec[:audio_length-1,:]
            save_audio(predicted_final_spectrogram_sample, self.params["logdir"], 0, mode="infer",number= i*batch_size+j)
          # if "spectrogram" in self.get_data_layer().params['output_type']:
          #   save_audio(predicted_final_spectrogram_sample, self.params["logdir"], 0, mode="infer", number=i*batch_size+j)

    return {}
