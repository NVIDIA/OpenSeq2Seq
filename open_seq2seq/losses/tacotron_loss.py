# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from .loss import Loss


class TacotronLoss(Loss):

  def __init__(self, params, model, name="tacotron_loss"):
    super(TacotronLoss, self).__init__(params, model, name)
    self._n_feats = self._model.get_data_layer().params['num_audio_features']
    if "both" in self._model.get_data_layer().params['output_type']:
      self._both = True
    else:
      self._both = False

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
        'scale': float,
    }

  def _compute_loss(self, input_dict):
    """Computes loss according to the tacotron 2 paper.

    Args:
      input_dict (dict): inputs to compute loss. Contains:

          * "decoder_output": dicionary containing:

              "outputs": array containing [

                  * decoder_predictions: spectrogram predicted by the decoder
                    rnn of shape [batch, time, feats]
                  * post_net_predictions: spectrogram after adding the residual
                    corrections from the post net of shape [batch, time, feats]
                  * stop_token_predictions: stop_token predictions of shape
                    [batch, time, 1]

              ]

          * "target_tensors": array containing [

              * spec: the true spectrogram of shape [batch, time, feats]
              * stop_token: the stop_token of shape [batch, time]

            ]

    Returns:
       Singleton loss tensor
    """
    # Compute loss in fp32
    decoder_predictions = input_dict['decoder_output']['outputs'][0]
    post_net_predictions = input_dict['decoder_output']['outputs'][1]
    stop_token_predictions = input_dict['decoder_output']['stop_token_prediction']
    if self._both:
      mag_pred = input_dict['decoder_output']['outputs'][5]
      mag_pred = tf.cast(mag_pred, dtype=tf.float32)
    spec = input_dict['target_tensors'][0]
    stop_token = input_dict['target_tensors'][1]
    stop_token = tf.expand_dims(stop_token, -1)
    spec_lengths = input_dict['target_tensors'][2]

    batch_size = tf.shape(spec)[0]
    num_feats = tf.shape(spec)[2]

    decoder_predictions = tf.cast(decoder_predictions, dtype=tf.float32)
    post_net_predictions = tf.cast(post_net_predictions, dtype=tf.float32)
    stop_token_predictions = tf.cast(stop_token_predictions, dtype=tf.float32)
    spec = tf.cast(spec, dtype=tf.float32)
    stop_token = tf.cast(stop_token, dtype=tf.float32)

    max_length = tf.to_int32(
        tf.maximum(
            tf.shape(spec)[1],
            tf.shape(decoder_predictions)[1],
        )
    )

    decoder_pad = tf.zeros(
        [
            batch_size,
            max_length - tf.shape(decoder_predictions)[1],
            tf.shape(decoder_predictions)[2]
        ]
    )
    stop_token_pred_pad = tf.zeros(
        [batch_size, max_length - tf.shape(decoder_predictions)[1], 1]
    )
    spec_pad = tf.zeros([batch_size, max_length - tf.shape(spec)[1], num_feats])
    stop_token_pad = tf.zeros([batch_size, max_length - tf.shape(spec)[1], 1])
    decoder_predictions = tf.concat(
        [decoder_predictions, decoder_pad], axis=1
    )
    post_net_predictions = tf.concat(
        [post_net_predictions, decoder_pad], axis=1
    )
    stop_token_predictions = tf.concat(
        [stop_token_predictions, stop_token_pred_pad], axis=1
    )
    spec = tf.concat([spec, spec_pad], axis=1)
    stop_token = tf.concat([stop_token, stop_token_pad], axis=1)

    if self._both:
      mag_pad = tf.zeros(
          [
              batch_size,
              max_length - tf.shape(mag_pred)[1],
              tf.shape(mag_pred)[2]
          ]
      )
      mag_pred = tf.concat(
         [mag_pred, mag_pad], axis=1
      )

      spec, mag_target = tf.split(
          spec,
          [self._n_feats['mel'], self._n_feats['magnitude']],
          axis=2
      )
    decoder_target = spec
    post_net_target = spec

    if self.params.get("use_mask", True):
      mask = tf.sequence_mask(
          lengths=spec_lengths, maxlen=max_length, dtype=tf.float32
      )
      mask = tf.expand_dims(mask, axis=-1)
      decoder_loss = tf.losses.mean_squared_error(
          labels=decoder_target, predictions=decoder_predictions, weights=mask
      )
      post_net_loss = tf.losses.mean_squared_error(
          labels=post_net_target, predictions=post_net_predictions, weights=mask
      )
      if self._both:
        mag_loss = tf.losses.mean_squared_error(
            labels=mag_target, predictions=mag_pred, weights=mask
        )
      stop_token_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=stop_token, logits=stop_token_predictions
      )
      stop_token_loss = stop_token_loss * mask
      stop_token_loss = tf.reduce_sum(stop_token_loss) / tf.reduce_sum(mask)

    else:
      decoder_loss = tf.losses.mean_squared_error(
          labels=decoder_target, predictions=decoder_predictions
      )
      post_net_loss = tf.losses.mean_squared_error(
          labels=post_net_target, predictions=post_net_predictions
      )
      if self._both:
        mag_loss = tf.losses.mean_squared_error(
           labels=mag_target, predictions=mag_pred
        )
      stop_token_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=stop_token, logits=stop_token_predictions
      )
      stop_token_loss = tf.reduce_mean(stop_token_loss)

    loss = decoder_loss + post_net_loss + stop_token_loss

    if self._both:
      loss += mag_loss

    if self.params.get("scale", None):
      loss = loss * self.params["scale"]
    return loss
