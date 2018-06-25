# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf

from .loss import Loss

class TacotronLoss(Loss):
  def __init__(self, params, model, name="cross_entropy_loss"):
    super(TacotronLoss, self).__init__(params, model, name)

  def get_optional_params(self):
    """Static method with description of optional parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **can** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return {
      'use_mask': bool,
    }

  def _compute_loss(self, input_dict):
    decoder_predictions = input_dict['decoder_output']['decoder_output']
    post_net_predictions = input_dict['decoder_output']['post_net_output']
    stop_token_predictions = input_dict['decoder_output']['target_output']
    spec = input_dict['target_tensors'][0]
    stop_token = input_dict['target_tensors'][1]
    stop_token = tf.expand_dims(stop_token, -1)
    spec_lengths = input_dict['target_tensors'][2]

    batch_size = tf.shape(spec)[0]
    num_feats = tf.shape(spec)[2]

    post_net_predictions = decoder_predictions + post_net_predictions

    predictions_pad = tf.zeros([batch_size, tf.shape(spec)[1]-tf.shape(decoder_predictions)[1],num_feats])
    stop_token_pad = tf.zeros([batch_size, tf.shape(spec)[1]-tf.shape(decoder_predictions)[1],1])
    decoder_predictions = tf.concat([decoder_predictions, predictions_pad], axis=1)
    post_net_predictions = tf.concat([post_net_predictions, predictions_pad], axis=1)
    stop_token_predictions = tf.concat([stop_token_predictions, stop_token_pad], axis=1)

    if self.params.get("use_mask", True):
      mask = tf.sequence_mask(lengths=spec_lengths,
                              dtype=tf.float32)
      mask = tf.expand_dims(mask, axis=-1)
      decoder_loss = tf.losses.mean_squared_error(labels=spec, predictions=decoder_predictions, weights=mask)
      post_net_loss = tf.losses.mean_squared_error(labels=spec, predictions=post_net_predictions, weights=mask)
      stop_token_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=stop_token, logits=stop_token_predictions)
      stop_token_loss = stop_token_loss * mask
      stop_token_loss = tf.reduce_sum(stop_token_loss) / tf.reduce_sum(mask)

    else:
      decoder_loss = tf.losses.mean_squared_error(labels=spec, predictions=decoder_predictions)
      post_net_loss = tf.losses.mean_squared_error(labels=spec, predictions=post_net_predictions)
      stop_token_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=stop_token, logits=stop_token_predictions)

    loss = decoder_loss + post_net_loss + stop_token_loss
    return loss