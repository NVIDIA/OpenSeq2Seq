# Copyright (c) 2018 NVIDIA Corporation

import tensorflow as tf

from .loss import Loss

class WavenetLoss(Loss):

  def __init__(self, params, model, name="wavenet_loss"):
    super(WavenetLoss, self).__init__(params, model, name)
    self._n_feats = self._model.get_data_layer().params["num_audio_features"]

  def get_required_params(self):
    return dict(Loss.get_required_params(), **{
      "receptive_field": int
    })

  def get_optional_params(self):
    return {}

  def _compute_loss(self, input_dict):
    receptive_field = self.params["receptive_field"]

    prediction = input_dict["decoder_output"]["logits"]
    target_output = input_dict["decoder_output"]["outputs"][0]

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=target_output)
    loss = tf.reduce_mean(loss)

    return loss
