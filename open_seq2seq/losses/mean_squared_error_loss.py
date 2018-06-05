# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf

from .loss import Loss


class MeanSquaredErrorLoss(Loss):
  def __init__(self, params, model, name="cross_entropy_loss"):
    super(MeanSquaredErrorLoss, self).__init__(params, model, name)

  def _compute_loss(self, input_dict):
    decoder_predictions = input_dict['decoder_output']['decoder_output']
    post_net_predictions = input_dict['decoder_output']['post_net_output']
    labels = input_dict['target_tensors'][0]
    tgt_lengths = input_dict['target_tensors'][1]

    batch_size = tf.shape(labels)[0]
    num_feats = tf.shape(labels)[2]

    post_net_predictions = decoder_predictions + post_net_predictions

    # pad_to = tf.to_int32(tf.maximum(
    #     tf.shape(labels)[1],
    #     tf.shape(predictions)[1],
    #   ))

    # pred_pad = tf.zeros([batch_size, tf.shape(predictions)[1], 2])
    # tf.shape(predictions)[1] - pad_to
    predictions_pad = tf.zeros([batch_size, tf.shape(labels)[1]-tf.shape(decoder_predictions)[1],num_feats])
    decoder_predictions = tf.concat([decoder_predictions, predictions_pad], axis=1)
    post_net_predictions = tf.concat([post_net_predictions, predictions_pad], axis=1)

    # current_ts = tf.to_int32(tf.minimum(
    #     tf.shape(labels)[1],
    #     tf.shape(predictions)[1],
    #   ))

    # predictions = tf.slice(
    #               predictions,
    #               begin=[0, 0, 0],
    #               size=[-1, current_ts, -1],
    # )
    # labels = tf.slice(labels,
    #                  begin=[0, 0, 0],
    #                  size=[-1, current_ts, -1])

    mask = tf.sequence_mask(lengths=tgt_lengths,
                              dtype=tf.float32)
    # mask = tf.expand_dims(mask, axis=-1)

    # print(mask.shape)
    # predictions = tf.boolean_mask(predictions, mask)
    # print(predictions.shape)
    # predictions = tf.reshape(predictions, [batch_size, -1, 96])

    decoder_loss = tf.losses.mean_squared_error(labels=labels, predictions=decoder_predictions)
    post_net_loss = tf.losses.mean_squared_error(labels=labels, predictions=post_net_predictions)
    loss = tf.reduce_sum((decoder_loss + post_net_loss) * mask)
    loss /= tf.reduce_sum(mask)
    # loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
    # loss = tf.clip_by_norm(loss, 1.)
    return loss

class BasicMeanSquaredErrorLoss(Loss):
  @staticmethod
  def get_optional_params():
    return dict(Loss.get_optional_params(), **{
      "output_key": str,
    })
    
  def __init__(self, params, model, name="cross_entropy_loss"):
    super(BasicMeanSquaredErrorLoss, self).__init__(params, model, name)

  def _compute_loss(self, input_dict):
    output_key = self.params.get('output_key', 'decoder_output')
    decoder_predictions = input_dict['decoder_output'][output_key]
    # post_net_predictions = input_dict['decoder_output']['post_net_output']
    labels = input_dict['target_tensors'][0]
    tgt_lengths = input_dict['target_tensors'][1]

    batch_size = tf.shape(labels)[0]
    num_feats = tf.shape(labels)[2]

    # pad_to = tf.to_int32(tf.maximum(
    #     tf.shape(labels)[1],
    #     tf.shape(predictions)[1],
    #   ))

    # pred_pad = tf.zeros([batch_size, tf.shape(predictions)[1], 2])
    # tf.shape(predictions)[1] - pad_to
    predictions_pad = tf.zeros([batch_size, tf.shape(labels)[1]-tf.shape(decoder_predictions)[1],num_feats])
    decoder_predictions = tf.concat([decoder_predictions, predictions_pad], axis=1)
    # post_net_predictions = tf.concat([post_net_predictions, predictions_pad], axis=1)

    # current_ts = tf.to_int32(tf.minimum(
    #     tf.shape(labels)[1],
    #     tf.shape(predictions)[1],
    #   ))

    # predictions = tf.slice(
    #               predictions,
    #               begin=[0, 0, 0],
    #               size=[-1, current_ts, -1],
    # )
    # labels = tf.slice(labels,
    #                  begin=[0, 0, 0],
    #                  size=[-1, current_ts, -1])

    mask = tf.sequence_mask(lengths=tgt_lengths,
                              dtype=tf.float32)
    # mask = tf.expand_dims(mask, axis=-1)

    # print(mask.shape)
    # predictions = tf.boolean_mask(predictions, mask)
    # print(predictions.shape)
    # predictions = tf.reshape(predictions, [batch_size, -1, 96])

    decoder_loss = tf.losses.mean_squared_error(labels=labels, predictions=decoder_predictions)
    # post_net_loss = tf.losses.mean_squared_error(labels=labels, predictions=post_net_predictions)
    loss = tf.reduce_sum(decoder_loss * mask)
    loss /= tf.reduce_sum(mask)
    # loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
    return loss
