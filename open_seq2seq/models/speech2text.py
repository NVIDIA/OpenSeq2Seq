# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import pandas as pd
import tensorflow as tf
import numpy as np

from .seq2seq import Seq2Seq
from open_seq2seq.utils.utils import deco_print


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


class Speech2Text(Seq2Seq):
  def maybe_print_logs(self, input_values, output_values):
    x, len_x, y, len_y = input_values
    decoded_sequence = output_values
    y_one_sample = y[0]
    len_y_one_sample = len_y[0]
    decoded_sequence_one_batch = decoded_sequence[0]

    # we also clip the sample by the correct length
    true_text = "".join(map(
      self.data_layer.params['idx2char'].get,
      y_one_sample[:len_y_one_sample],
    ))
    pred_text = "".join(sparse_tensor_to_chars(
      decoded_sequence_one_batch, self.data_layer.params['idx2char'])[0]
    )
    sample_wer = levenshtein(true_text.split(), pred_text.split()) / \
                 len(true_text.split())

    deco_print("Sample WER: {:.4f}".format(sample_wer), offset=4)
    deco_print("Sample target:     " + true_text, offset=4)
    deco_print("Sample prediction: " + pred_text, offset=4)
    return {
      'Sample WER': sample_wer,
    }

  def clip_last_batch(self, last_batch, true_size):
    def clip_sparse(value, size):
      dense_shape_clipped = value.dense_shape
      dense_shape_clipped[0] = size
      indices_clipped = []
      values_clipped = []
      for idx_tuple, val in zip(value.indices, value.values):
        if idx_tuple[0] < size:
          indices_clipped.append(idx_tuple)
          values_clipped.append(val)
      return tf.SparseTensorValue(np.array(indices_clipped),
                                  np.array(values_clipped),
                                  dense_shape_clipped)

    last_batch_clipped = []
    for val in last_batch:
      if isinstance(val, tf.SparseTensorValue):
        last_batch_clipped.append(clip_sparse(val, true_size))
      else:
        last_batch_clipped.append(val[:true_size])
    return last_batch_clipped

  def maybe_evaluate(self, inputs_per_batch, outputs_per_batch):
    total_word_lev = 0.0
    total_word_count = 0.0

    for input_values, output_values in zip(inputs_per_batch, outputs_per_batch):
      decoded_sequence = output_values[0]
      decoded_texts = sparse_tensor_to_chars(
        decoded_sequence,
        self.data_layer.params['idx2char'],
      )
      for sample_id in range(input_values[0].shape[0]):
        # y is the third returned input value, thus input_values[2]
        # len_y is the fourth returned input value
        y = input_values[2][sample_id]
        len_y = input_values[3][sample_id]
        true_text = "".join(map(self.data_layer.params['idx2char'].get,
                                y[:len_y]))
        pred_text = "".join(decoded_texts[sample_id])

        total_word_lev += levenshtein(true_text.split(), pred_text.split())
        total_word_count += len(true_text.split())

    total_wer = 1.0 * total_word_lev / total_word_count
    deco_print("Validation WER:  {:.4f}".format(total_wer), offset=4)
    return {
      "Eval WER": total_wer,
    }

  def infer(self, inputs_per_batch, outputs_per_batch, output_file):
    preds = []
    samples_count = 0
    dataset_size = self.data_layer.get_size_in_samples()

    for input_values, output_values in zip(inputs_per_batch,
                                           outputs_per_batch):
      for gpu_id in range(self.num_gpus):
        decoded_sequence = output_values[gpu_id]
        decoded_texts = sparse_tensor_to_chars(
          decoded_sequence,
          self.data_layer.params['idx2char'],
        )
        for sample_id in range(self.params['batch_size_per_gpu']):
          # this is necessary for correct processing of the last batch
          if samples_count >= dataset_size:
            break
          samples_count += 1
          preds.append("".join(decoded_texts[sample_id]))
    pd.DataFrame(
      {
        'wav_filename': self.data_layer.params['files'],
        'predicted_transcript': preds,
      },
      columns=['wav_filename', 'predicted_transcript'],
    ).to_csv(output_file, index=False)
