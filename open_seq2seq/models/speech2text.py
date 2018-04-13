# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Original work Copyright (c) 2018 Mozilla Corporation
# Modified work Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from .seq2seq import Seq2Seq
from open_seq2seq.utils import deco_print
import pandas as pd


def sparse_tuple_to_texts(tup, alphabet):
  indices = tup[0]
  values = tup[1]
  results = [''] * tup[2][0]
  for i in range(len(indices)):
    index = indices[i][0]
    results[index] += alphabet.string_from_label(values[i])
  # List of strings
  return results


def sparse_tensor_value_to_texts(value, alphabet):
  r"""
  Given a :class:`tf.SparseTensor` ``value``, return an array of
  Python strings representing its values.
  """
  return sparse_tuple_to_texts((value.indices, value.values,
                                value.dense_shape), alphabet)


# The following code is from: http://hetland.org/coding/python/levenshtein.py

# This is a straightforward implementation of a well-known algorithm, and thus
# probably shouldn't be covered by copyright to begin with. But in case it is,
# the author (Magnus Lie Hetland) has, to the extent possible under law,
# dedicated all copyright and related and neighboring rights to this software
# to the public domain worldwide, by distributing it under the CC0 license,
# version 1.0. This software is distributed without any warranty. For more
# information, see <http://creativecommons.org/publicdomain/zero/1.0>
def levenshtein(a, b):
    """Calculates the Levenshtein distance between a and b."""
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
    # using only the first sample from the batch on the first gpu, thus y[0][0]
    if self.on_horovod:
      y_one_sample = y[0]
      len_y_one_sample = len_y[0]
      decoded_sequence_one_batch = decoded_sequence[0]
    else:
      y_one_sample = y[0][0]
      len_y_one_sample = len_y[0][0]
      decoded_sequence_one_batch = decoded_sequence[0]

    # we also clip the sample by the correct length
    true_text = "".join(map(
      self.data_layer.params['alphabet'].string_from_label,
      y_one_sample[:len_y_one_sample],
    ))
    pred_text = "".join(sparse_tensor_value_to_texts(
      decoded_sequence_one_batch, self.data_layer.params['alphabet'])[0]
    )
    sample_med = levenshtein(true_text, pred_text) / len(true_text)
    sample_wer = levenshtein(true_text.split(), pred_text.split()) / \
                 len(true_text.split())

    # deco_print("Sample mean edit distance: {}".format(sample_med), offset=4)
    deco_print("Sample WER: {:.4f}".format(sample_wer), offset=4)
    deco_print("Sample target:     " + true_text, offset=4)
    deco_print("Sample prediction: " + pred_text, offset=4)
    return {
      'Sample MED': sample_med,
      'Sample WER': sample_wer,
    }

  def maybe_evaluate(self, inputs_per_batch, outputs_per_batch):
    total_char_lev = 0.0
    total_word_lev = 0.0
    total_word_count = 0.0
    total_char_count = 0.0

    for input_values, output_values in zip(inputs_per_batch, outputs_per_batch):
      for gpu_id in range(self._num_gpus):
        decoded_sequence = output_values[gpu_id]
        decoded_texts = sparse_tensor_value_to_texts(
          decoded_sequence,
          self.data_layer.params['alphabet'],
        )
        for sample_id in range(self.params['batch_size_per_gpu']):
          # y is the third returned input value, thus input_values[2]
          # len_y is the fourth returned input value
          y = input_values[2][gpu_id][sample_id]
          len_y = input_values[3][gpu_id][sample_id]
          true_text = "".join(map(
            self.data_layer.params['alphabet'].string_from_label,
            y[:len_y],
          ))
          pred_text = "".join(decoded_texts[sample_id])

          total_char_lev += levenshtein(true_text, pred_text)
          total_word_lev += levenshtein(true_text.split(), pred_text.split())
          total_word_count += len(true_text.split())
          total_char_count += len(true_text)

    total_wer = 1.0 * total_word_lev / total_word_count
    total_med = 1.0 * total_char_lev / total_char_count

    # deco_print("Validation mean edit distance: {}".format(total_med), offset=4)
    deco_print("Validation WER:  {:.4f}".format(total_wer), offset=4)
    return {
      "Eval MED": total_med,
      "Eval WER": total_wer,
    }

  def infer(self, inputs_per_batch, outputs_per_batch, output_file):
    preds = []
    for input_values, output_values in zip(inputs_per_batch,
                                           outputs_per_batch):
      for gpu_id in range(self._num_gpus):
        decoded_sequence = output_values[gpu_id]
        decoded_texts = sparse_tensor_value_to_texts(
          decoded_sequence,
          self.data_layer.params['alphabet'],
        )
        for sample_id in range(self.params['batch_size_per_gpu']):
          preds.append("".join(decoded_texts[sample_id]))
    pd.DataFrame(
      {
        'wav_filename': self.data_layer.params['files'],
        'predicted_transcript': preds,
      },
      columns=['wav_filename', 'predicted_transcript'],
    ).to_csv(output_file, index=False)
