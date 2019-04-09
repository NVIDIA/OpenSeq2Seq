# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from six.moves import range
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

from open_seq2seq.utils.utils import deco_print
from .encoder_decoder import EncoderDecoderModel

import pickle


def sparse_tensor_to_chars(tensor, idx2char):
  text = [''] * tensor.dense_shape[0]
  for idx_tuple, value in zip(tensor.indices, tensor.values):
    text[idx_tuple[0]] += idx2char[value]
  return text


def sparse_tensor_to_chars_bpe(tensor):
  idx = [[] for _ in range(tensor.dense_shape[0])]
  for idx_tuple, value in zip(tensor.indices, tensor.values):
    idx[idx_tuple[0]].append(int(value))
  
  return idx


def dense_tensor_to_chars(tensor, idx2char, startindex, endindex):
  batch_size = len(tensor)
  text = [''] * batch_size
  for batch_num in range(batch_size):
    '''text[batch_num] = "".join([idx2char[idx] for idx in tensor[batch_num]
                               if idx not in [startindex, endindex]])'''

    text[batch_num] = ""
    for idx in tensor[batch_num]:
      if idx == endindex:
        break
      text[batch_num] += idx2char[idx]
  return text


def levenshtein(a, b):
  """Calculates the Levenshtein distance between a and b.
  The code was copied from: http://hetland.org/coding/python/levenshtein.py
  """
  n, m = len(a), len(b)
  if n > m:
    # Make sure n <= m, to use O(min(n,m)) space
    a, b = b, a
    n, m = m, n

  current = list(range(n + 1))
  for i in range(1, m + 1):
    previous, current = current, [i] + [0] * n
    for j in range(1, n + 1):
      add, delete = previous[j] + 1, current[j - 1] + 1
      change = previous[j - 1]
      if a[j - 1] != b[i - 1]:
        change = change + 1
      current[j] = min(add, delete, change)

  return current[n]


def plot_attention(alignments, pred_text, encoder_len, training_step):

  alignments = alignments[:len(pred_text), :encoder_len]
  fig = plt.figure(figsize=(15, 10))
  ax = fig.add_subplot(1, 1, 1)

  img = ax.imshow(alignments, interpolation='nearest', cmap='Blues')
  ax.grid()
  #fig.savefig('/home/rgadde/Desktop/OpenSeq2Seq/plots/file{}.png'.format(training_step), dpi=300)

  sbuffer = BytesIO()
  fig.savefig(sbuffer, dpi=300)
  summary = tf.Summary.Image(
      encoded_image_string=sbuffer.getvalue(),
      height=int(fig.get_figheight() * 2),
      width=int(fig.get_figwidth() * 2)
  )
  summary = tf.Summary.Value(
      tag="attention_summary_step_{}".format(int(training_step / 2200)), image=summary)

  plt.close(fig)
  return summary


class Speech2Text(EncoderDecoderModel):

  def _create_decoder(self):
    data_layer = self.get_data_layer()
    self.params['decoder_params']['tgt_vocab_size'] = (
        data_layer.params['tgt_vocab_size']
    )

    self.dump_outputs = self.params['decoder_params'].get('infer_logits_to_pickle', False)

    self.is_bpe = data_layer.params.get('bpe', False)
    self.tensor_to_chars = sparse_tensor_to_chars
    self.tensor_to_char_params = {}
    self.autoregressive = data_layer.params.get('autoregressive', False)
    if self.autoregressive:
      self.params['decoder_params']['GO_SYMBOL'] = data_layer.start_index
      self.params['decoder_params']['END_SYMBOL'] = data_layer.end_index
      self.tensor_to_chars = dense_tensor_to_chars
      self.tensor_to_char_params['startindex'] = data_layer.start_index
      self.tensor_to_char_params['endindex'] = data_layer.end_index

    return super(Speech2Text, self)._create_decoder()

  def _create_loss(self):
    if self.get_data_layer().params.get('autoregressive', False):
      self.params['loss_params'][
          'batch_size'] = self.params['batch_size_per_gpu']
      self.params['loss_params']['tgt_vocab_size'] = (
          self.get_data_layer().params['tgt_vocab_size']
      )
    return super(Speech2Text, self)._create_loss()

  def _build_forward_pass_graph(self, input_tensors, gpu_id=0):
    """TensorFlow graph for speech2text model is created here.
    This function connects encoder, decoder and loss together. As an input for
    encoder it will specify source tensors (as returned from
    the data layer). As an input for decoder it will specify target tensors
    as well as all output returned from encoder. For loss it
    will also specify target tensors and all output returned from
    decoder. Note that loss will only be built for mode == "train" or "eval".

    Args:
      input_tensors (dict): ``input_tensors`` dictionary that has to contain
          ``source_tensors`` key with the list of all source tensors, and
          ``target_tensors`` with the list of all target tensors. Note that
          ``target_tensors`` only need to be provided if mode is
          "train" or "eval".
      gpu_id (int, optional): id of the GPU where the current copy of the model
          is constructed. For Horovod this is always zero.

    Returns:
      tuple: tuple containing loss tensor as returned from
      ``loss.compute_loss()`` and list of outputs tensors, which is taken from
      ``decoder.decode()['outputs']``. When ``mode == 'infer'``, loss will
      be None.
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
      if self.mode == "train" or self.mode == "eval":
        decoder_input['target_tensors'] = target_tensors
      decoder_output = self.decoder.decode(input_dict=decoder_input)
      model_outputs = decoder_output.get("outputs", None)

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
        if self.dump_outputs:
          model_logits = decoder_output.get("logits", None)
          return loss, [model_logits]
    return loss, model_outputs



  def maybe_print_logs(self, input_values, output_values, training_step):
    y, len_y = input_values['target_tensors']
    decoded_sequence = output_values
    y_one_sample = y[0]
    len_y_one_sample = len_y[0]
    decoded_sequence_one_batch = decoded_sequence[0]

    if self.is_bpe:
      dec_list = sparse_tensor_to_chars_bpe(decoded_sequence_one_batch)[0]
      true_text = self.get_data_layer().sp.DecodeIds(y_one_sample[:len_y_one_sample].tolist())
      pred_text = self.get_data_layer().sp.DecodeIds(dec_list)

    else:
      # we also clip the sample by the correct length
      true_text = "".join(map(
          self.get_data_layer().params['idx2char'].get,
          y_one_sample[:len_y_one_sample],
      ))
      pred_text = "".join(self.tensor_to_chars(
          decoded_sequence_one_batch,
          self.get_data_layer().params['idx2char'],
          **self.tensor_to_char_params
      )[0])
    sample_wer = levenshtein(true_text.split(), pred_text.split()) / \
        len(true_text.split())

    self.autoregressive = self.get_data_layer().params.get('autoregressive', False)
    self.plot_attention = False  # (output_values[1] != None).all()
    if self.plot_attention:
      attention_summary = plot_attention(
          output_values[1][0], pred_text, output_values[2][0], training_step)

    deco_print("Sample WER: {:.4f}".format(sample_wer), offset=4)
    deco_print("Sample target:     " + true_text, offset=4)
    deco_print("Sample prediction: " + pred_text, offset=4)

    if self.plot_attention:
      return {
          'Sample WER': sample_wer,
          'Attention Summary': attention_summary,
      }
    else:
      return {
          'Sample WER': sample_wer,
      }
    
  def finalize_evaluation(self, results_per_batch, training_step=None):
    total_word_lev = 0.0
    total_word_count = 0.0
    for word_lev, word_count in results_per_batch:
      total_word_lev += word_lev
      total_word_count += word_count

    total_wer = 1.0 * total_word_lev / total_word_count
    deco_print("Validation WER:  {:.4f}".format(total_wer), offset=4)
    return {
        "Eval WER": total_wer,
    }

  def evaluate(self, input_values, output_values):
    total_word_lev = 0.0
    total_word_count = 0.0

    decoded_sequence = output_values[0]

    if self.is_bpe:
      decoded_texts = sparse_tensor_to_chars_bpe(decoded_sequence)
    else:
      decoded_texts = self.tensor_to_chars(
          decoded_sequence,
          self.get_data_layer().params['idx2char'],
          **self.tensor_to_char_params
      )

    batch_size = input_values['source_tensors'][0].shape[0]
    for sample_id in range(batch_size):
      # y is the third returned input value, thus input_values[2]
      # len_y is the fourth returned input value
      y = input_values['target_tensors'][0][sample_id]
      len_y = input_values['target_tensors'][1][sample_id]
      if self.is_bpe:
        true_text = self.get_data_layer().sp.DecodeIds(y[:len_y].tolist())
        pred_text = self.get_data_layer().sp.DecodeIds(decoded_texts[sample_id])
      else:
        true_text = "".join(map(self.get_data_layer().params['idx2char'].get,
                              y[:len_y]))
        pred_text = "".join(decoded_texts[sample_id])
      if self.get_data_layer().params.get('autoregressive', False):
        true_text = true_text[:-4]

      # print('TRUE_TEXT: "{}"'.format(true_text))
      # print('PRED_TEXT: "{}"'.format(pred_text))

      total_word_lev += levenshtein(true_text.split(), pred_text.split())
      total_word_count += len(true_text.split())

    return total_word_lev, total_word_count

  def infer(self, input_values, output_values):
    preds = []
    decoded_sequence = output_values[0]

    if self.dump_outputs:
      # decoded_sequence has 'time_major' shape: [T, B, C]
      for i in range(decoded_sequence.shape[0]):
        preds.append(decoded_sequence[i, :, :].squeeze())
    else:
      decoded_texts = self.tensor_to_chars(
          decoded_sequence,
          self.get_data_layer().params['idx2char'],
          **self.tensor_to_char_params
      )
      for decoded_text in decoded_texts:
        preds.append("".join(decoded_text))

    return preds, input_values['source_ids']

  def finalize_inference(self, results_per_batch, output_file):
    preds = []
    ids = []

    for result, idx in results_per_batch:
      preds.extend(result)
      ids.extend(idx)

    preds = np.array(preds)
    ids = np.hstack(ids)
    # restoring the correct order
    preds = preds[np.argsort(ids)]
    if self.dump_outputs:
      dump_out = {}
      dump_results = {}
      files = self.get_data_layer().all_files
      for i, f in enumerate(files):
        dump_results[f] = preds[i]
      dump_out["logits"] = dump_results
      step_size = self.get_data_layer().params["window_stride"]
      scale = 1
      # check strides in convolutional layers
      for layers in ['convnet_layers', 'conv_layers', 'cnn_layers']:
        convs = self.encoder.params.get(layers)
        if convs:
          for c in convs:
            scale *= c["stride"][0]
      dump_out["step_size"] = scale*step_size
      dump_out["vocab"] = self.get_data_layer().params['idx2char']
      with open(output_file, 'wb') as f:
        pickle.dump(dump_out, f, protocol=pickle.HIGHEST_PROTOCOL)
      f.close()
    else:
      pd.DataFrame(
          {
              'wav_filename': self.get_data_layer().all_files,
              'predicted_transcript': preds,
          },
          columns=['wav_filename', 'predicted_transcript'],
      ).to_csv(output_file, index=False)

  def _get_num_objects_per_step(self, worker_id=0):
    """Returns number of audio frames in current batch."""
    data_layer = self.get_data_layer(worker_id)
    num_frames = tf.reduce_sum(data_layer.input_tensors['source_tensors'][1])
    return num_frames
