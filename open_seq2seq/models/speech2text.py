# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from six.moves import range

from open_seq2seq.utils.utils import deco_print
from .encoder_decoder import EncoderDecoderModel


def sparse_tensor_to_chars(tensor, idx2char):
  text = [''] * tensor.dense_shape[0]
  for idx_tuple, value in zip(tensor.indices, tensor.values):
    text[idx_tuple[0]] += idx2char[value]
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


class Speech2Text(EncoderDecoderModel):
  def _create_decoder(self):
    self.params['decoder_params']['tgt_vocab_size'] = (
        self.get_data_layer().params['tgt_vocab_size']
    )
    return super(Speech2Text, self)._create_decoder()

  def maybe_print_logs(self, input_values, output_values, training_step):
    y, len_y = input_values['target_tensors']
    decoded_sequence = output_values
    y_one_sample = y[0]
    len_y_one_sample = len_y[0]
    decoded_sequence_one_batch = decoded_sequence[0]

    # we also clip the sample by the correct length
    true_text = "".join(map(
        self.get_data_layer().params['idx2char'].get,
        y_one_sample[:len_y_one_sample],
    ))
    pred_text = "".join(sparse_tensor_to_chars(
        decoded_sequence_one_batch,
        self.get_data_layer().params['idx2char']
    )[0])
    sample_wer = levenshtein(true_text.split(), pred_text.split()) / \
                 len(true_text.split())

    deco_print("Sample WER: {:.4f}".format(sample_wer), offset=4)
    deco_print("Sample target:     " + true_text, offset=4)
    deco_print("Sample prediction: " + pred_text, offset=4)
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
    decoded_texts = sparse_tensor_to_chars(
        decoded_sequence,
        self.get_data_layer().params['idx2char'],
    )
    batch_size = input_values['source_tensors'][0].shape[0]
    for sample_id in range(batch_size):
      # y is the third returned input value, thus input_values[2]
      # len_y is the fourth returned input value
      y = input_values['target_tensors'][0][sample_id]
      len_y = input_values['target_tensors'][1][sample_id]
      true_text = "".join(map(self.get_data_layer().params['idx2char'].get,
                              y[:len_y]))
      pred_text = "".join(decoded_texts[sample_id])

      total_word_lev += levenshtein(true_text.split(), pred_text.split())
      total_word_count += len(true_text.split())

    return total_word_lev, total_word_count

  def infer(self, input_values, output_values):
    preds = []
    decoded_sequence = output_values[0]
    decoded_texts = sparse_tensor_to_chars(
        decoded_sequence,
        self.get_data_layer().params['idx2char'],
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
