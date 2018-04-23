# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Original work Copyright (c) 2018 Mozilla Corporation
# Modified work Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from functools import reduce

from .loss import Loss
from open_seq2seq.utils.utils import mask_nans, deco_print


def gather_nd(params, indices, shape):
  rank = len(shape)
  flat_params = tf.reshape(params, [-1])
  multipliers = [reduce(lambda x, y: x * y, shape[i + 1:], 1)
                 for i in range(0, rank)]
  indices_unpacked = tf.unstack(tf.transpose(
    indices, [rank - 1] + list(range(0, rank - 1)),
  ))
  flat_indices = sum([a * b for a, b in zip(multipliers, indices_unpacked)])
  return tf.gather(flat_params, flat_indices)


def ctc_label_dense_to_sparse(labels, label_lengths, batch_size):
  # The second dimension of labels must be equal to the
  # longest label length in the batch
  correct_shape_assert = tf.assert_equal(
    tf.shape(labels)[1], tf.reduce_max(label_lengths),
  )
  with tf.control_dependencies([correct_shape_assert]):
    labels = tf.identity(labels)

  label_shape = tf.shape(labels)
  num_batches_tns = tf.stack([label_shape[0]])
  max_num_labels_tns = tf.stack([label_shape[1]])

  def range_less_than(previous_state, current_input):
    return tf.expand_dims(tf.range(label_shape[1]), 0) < current_input

  init = tf.cast(tf.fill(max_num_labels_tns, 0), tf.bool)
  init = tf.expand_dims(init, 0)
  dense_mask = tf.scan(
    range_less_than, label_lengths,
    initializer=init, parallel_iterations=1,
  )
  dense_mask = dense_mask[:, 0, :]

  label_array = tf.reshape(
    tf.tile(tf.range(0, label_shape[1]), num_batches_tns), label_shape,
  )
  label_ind = tf.boolean_mask(label_array, dense_mask)

  batch_array = tf.transpose(tf.reshape(
    tf.tile(tf.range(0, label_shape[0]), max_num_labels_tns),
    tf.reverse(label_shape, [0]),
  ))
  batch_ind = tf.boolean_mask(batch_array, dense_mask)

  indices = tf.transpose(
    tf.reshape(tf.concat([batch_ind, label_ind], 0), [2, -1]),
  )
  shape = [batch_size, tf.reduce_max(label_lengths)]
  vals_sparse = gather_nd(labels, indices, shape)

  return tf.SparseTensor(tf.to_int64(indices), vals_sparse,
                         tf.to_int64(label_shape))


class CTCLoss(Loss):
  """Implementation of the CTC loss."""
  @staticmethod
  def get_optional_params():
    return dict(Loss.get_optional_params(), **{
      'mask_nan': bool,
    })

  def __init__(self, params, model, name="ctc_loss"):
    super(CTCLoss, self).__init__(params, model, name)
    self._mask_nan = self.params.get("mask_nan", True)
    # this loss can only operate in full precision
    if self.params['dtype'] != tf.float32:
      deco_print("Warning: defaulting ctc loss to work in float32")
    self.params['dtype'] = tf.float32

  def _compute_loss(self, input_dict):
    """
    Computes CTC loss
    :param input_dict: inputs to compute loss
    {
          "logits": logits tensor of shape [batch_size, T, dim]
          "target_sequence": tensor of shape [batch_size, T]
          "src_lengths": tensor of shape [batch_size]
          "tgt_lengths": tensor of shape [batch_size]
    }
    :return: Singleton loss tensor
    """
    logits = input_dict['decoder_output']['logits']
    tgt_sequence = input_dict['tgt_sequence']
    tgt_length = input_dict['tgt_length']
    # this loss needs an access to src_length since they
    # might get changed in the encoder
    src_length = input_dict['decoder_output']['src_length']

    batch_size = tgt_length.shape.as_list()[0]

    # Converting targets to sparse tensor
    tgt_sequence = ctc_label_dense_to_sparse(
      tgt_sequence, tgt_length, batch_size,
    )

    # Compute the CTC loss
    total_loss = tf.nn.ctc_loss(
      labels=tgt_sequence,
      inputs=logits,
      sequence_length=src_length,
      ignore_longer_outputs_than_inputs=True,
    )

    if self._mask_nan:
      total_loss = mask_nans(total_loss)

    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(total_loss)
    return avg_loss
