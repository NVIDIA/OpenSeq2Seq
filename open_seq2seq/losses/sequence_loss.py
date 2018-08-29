# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from .loss import Loss


class BasicSequenceLoss(Loss):
  """
  Basic sequence-to-sequence loss. This one does not use one-hot encodings
  """
  @staticmethod
  def get_required_params():
    return dict(Loss.get_required_params(), **{
        'tgt_vocab_size': int,
        'batch_size': int,
    })

  @staticmethod
  def get_optional_params():
    return dict(Loss.get_optional_params(), **{
        'offset_target_by_one': bool,
        'average_across_timestep': bool,
        'do_mask': bool,
    })

  def __init__(self, params, model, name="basic_sequence_loss"):
    """Constructor.

    Args:
      params (dict): dictionary with loss parameters.
        Should contain the following:
        * tgt_vocab_size: Target vocabulary size
        * batch_size_per_gpu: Size of the per-worker batch
        * offset_target_by_one: (default: True). Keep it true for
        auto-regressive models
        * average_across_timestep: (default: False). If True, will average
          loss across timesteps, else it will sum across timesteps
        * do_mask: (default: True) whether to mask based on tgt_lengths
          (which is passed as part of loss_input_dict to compute_loss
          and has to be not None then)
    """
    super(BasicSequenceLoss, self).__init__(params, model, name)
    self._tgt_vocab_size = self.params["tgt_vocab_size"]
    self._batch_size = self.params["batch_size"]
    self._offset_target_by_one = self.params.get("offset_target_by_one", True)
    self._average_across_timestep = self.params.get("average_across_timestep",
                                                    False)
    self._do_mask = self.params.get("do_mask", True)

  def _compute_loss(self, input_dict):
    """Computes cross entropy based sequence-to-sequence loss.

    Args:
      input_dict (dict): inputs to compute loss::
        {
              "logits": logits tensor of shape [batch_size, T, dim]
              "target_sequence": tensor of shape [batch_size, T]
              "tgt_lengths": tensor of shape [batch_size] or None
        }

    Returns:
       Singleton loss tensor
    """
    logits = input_dict["decoder_output"]["logits"]
    target_sequence = input_dict['target_tensors'][0]
    tgt_lengths = input_dict['target_tensors'][1]

    if self._offset_target_by_one:
      # this is necessary for auto-regressive models
      current_ts = tf.to_int32(tf.minimum(
          tf.shape(target_sequence)[1],
          tf.shape(logits)[1],
      )) - 1

      logits = tf.slice(
          logits,
          begin=[0, 0, 0],
          size=[-1, current_ts, -1],
      )                                 
      target_sequence = tf.slice(target_sequence,
                                 begin=[0, 1],
                                 size=[-1, current_ts])
    else:
      current_ts = tf.to_int32(tf.minimum(
          tf.shape(target_sequence)[1],
          tf.shape(logits)[1],
      ))

    # Cast logits after potential slice
    if logits.dtype.base_dtype != tf.float32:
      logits = tf.cast(logits, tf.float32)

    if self._do_mask:
      if tgt_lengths is None:
        raise ValueError("If you are masking loss, tgt_lengths can't be None")
      mask = tf.sequence_mask(lengths=tgt_lengths - 1,
                              maxlen=current_ts,
                              dtype=logits.dtype)
    else:
      mask = tf.cast(tf.ones_like(target_sequence), logits.dtype)

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.reshape(target_sequence, shape=[-1]),
        logits=tf.reshape(logits, shape=[-1, self._tgt_vocab_size]),
    )
    if self._average_across_timestep:
      loss = tf.reduce_mean(crossent * tf.reshape(mask, shape=[-1]))
    else:
      loss = tf.reduce_sum(crossent * tf.reshape(mask, shape=[-1]))
      loss /= self._batch_size
    return loss


class CrossEntropyWithSmoothing(Loss):
  """Softmax cross entropy loss with label smoothing.
  This one uses one-hot encodings for labels.
  """
  @staticmethod
  def get_required_params():
    return dict(Loss.get_required_params(), **{
        'tgt_vocab_size': int,
        'batch_size': int,
    })

  @staticmethod
  def get_optional_params():
    return dict(Loss.get_optional_params(), **{
        'offset_target_by_one': bool,
        'average_across_timestep': bool,
        'do_mask': bool,
        'label_smoothing': float,
    })

  def __init__(self, params, model, name="cross_entropy_with_smoothing"):
    """Constructor.

    Args:
      params (dict): dictionary with loss parameters.
        Should contain the following:
          * tgt_vocab_size: Target vocabulary size
          * batch_size_per_gpu: Size of the per-worker batch
          * offset_target_by_one: (default: True). Keep it true for
            auto-regressive models
          * do_mask: (default: True) whether to mask based on tgt_lengths
            (which is passed as part of loss_input_dict to compute_loss
            and has to be not None then)
    """
    super(CrossEntropyWithSmoothing, self).__init__(params, model, name)
    self._tgt_vocab_size = self.params["tgt_vocab_size"]
    self._batch_size = self.params["batch_size"]
    self._offset_target_by_one = self.params.get("offset_target_by_one", True)
    self._do_mask = self.params.get("do_mask", True)
    self._label_smoothing = self.params.get("label_smoothing", 0.0)
    self._average_across_timestep = self.params.get("average_across_timestep",
                                                    False)

  def _compute_loss(self, input_dict):
    """Computes cross entropy based sequence-to-sequence loss
    with label smoothing.

    Args:
      input_dict (dict): inputs to compute loss::
        {
            "logits": logits tensor of shape [batch_size, T, dim]
            "target_sequence": tensor of shape [batch_size, T]
            "tgt_lengths": tensor of shape [batch_size] or None
        }
    Returns:
      Singleton loss tensor
    """
    logits = input_dict["decoder_output"]["logits"]
    target_sequence = input_dict["target_tensors"][0]
    tgt_lengths = input_dict["target_tensors"][1]

    if self._offset_target_by_one:
      # this is necessary for auto-regressive models      
      current_ts = tf.to_int32(tf.minimum(
          tf.shape(target_sequence)[1],
          tf.shape(logits)[1],
      )) - 1

      logits = tf.slice(
          logits,
          begin=[0, 0, 0],
          size=[-1, current_ts, -1],
      )
      target_sequence = tf.slice(target_sequence,
                                 begin=[0, 1],
                                 size=[-1, current_ts])
    else:
      current_ts = tf.to_int32(tf.minimum(
          tf.shape(target_sequence)[1],
          tf.shape(logits)[1],
      ))

    # Cast logits after potential slice
    if logits.dtype.base_dtype != tf.float32:
      logits = tf.cast(logits, tf.float32)

    if self._do_mask:
      if tgt_lengths is None:
        raise ValueError("If you are masking loss, tgt_lengths can't be None")
      mask = tf.sequence_mask(lengths=tgt_lengths - 1,
                              maxlen=current_ts,
                              dtype=tf.float32)
    else:
      mask = tf.cast(tf.ones_like(target_sequence), logits.dtype)

    labels = tf.one_hot(indices=tf.reshape(target_sequence, shape=[-1]),
                        depth=self._tgt_vocab_size)
    logits = tf.reshape(logits, shape=[-1, self._tgt_vocab_size])
    mask = tf.reshape(mask, shape=[-1])

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels,
        logits=logits,
        weights=mask,
        label_smoothing=self._label_smoothing,
        reduction=tf.losses.Reduction.NONE,
    )

    loss = tf.reduce_sum(loss * tf.reshape(mask, shape=[-1]))
    if self._average_across_timestep:
      loss /= tf.reduce_sum(mask)
    else:
      loss /= self._batch_size
    return loss


class PaddedCrossEntropyLossWithSmoothing(Loss):
  @staticmethod
  def get_optional_params():
    return dict(Loss.get_optional_params(), **{
        'batch_size': int,
        'tgt_vocab_size': int,
        'label_smoothing': float,
        'pad_embeddings_2_eight': bool,
    })

  def __init__(self, params, model, name="padded_cross_entropy_with_smoothing"):
    super(PaddedCrossEntropyLossWithSmoothing, self).__init__(params, model,
                                                              name)
    if self.params.get('pad_embeddings_2_eight', False):
      if self.params["tgt_vocab_size"] % 8 == 0:
        self._tgt_vocab_size = self.params["tgt_vocab_size"]
      else:
        self._tgt_vocab_size = self.params["tgt_vocab_size"] + \
                               (8 - self.params["tgt_vocab_size"] % 8)
    else:
      self._tgt_vocab_size = self.params["tgt_vocab_size"]
    self._label_smoothing = self.params.get("label_smoothing", 0.0)

  def _compute_loss(self, input_dict):
    logits = input_dict["decoder_output"]["logits"]
    logits = tf.cast(logits, dtype=tf.float32)
    if logits is None:
      return 0.0
    labels = input_dict["target_tensors"][0]

    def _pad_tensors_to_same_length(x, y):
      """Pad x and y so that the results have the
      same length (second dimension).
      """
      with tf.name_scope("pad_to_same_length"):
        x_length = tf.shape(x)[1]
        y_length = tf.shape(y)[1]

        max_length = tf.maximum(x_length, y_length)

        x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
        y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
        return x, y

    with tf.name_scope("loss", [logits, labels]):
      logits, labels = _pad_tensors_to_same_length(logits, labels)

      # Calculate smoothing cross entropy
      with tf.name_scope("smoothing_cross_entropy", [logits, labels]):
        confidence = 1.0 - self._label_smoothing
        low_confidence = (1.0 - confidence) / \
                         tf.to_float(self._tgt_vocab_size - 1)
        soft_targets = tf.one_hot(
            tf.cast(labels, tf.int32),
            depth=self._tgt_vocab_size,
            on_value=confidence,
            off_value=low_confidence,
        )
        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=soft_targets,
        )

        # Calculate the best (lowest) possible value of cross entropy, and
        # subtract from the cross entropy loss.
        normalizing_constant = -(
            confidence * tf.log(confidence) +
            tf.to_float(self._tgt_vocab_size - 1) *
            low_confidence * tf.log(low_confidence + 1e-20)
        )
        xentropy -= normalizing_constant

      weights = tf.to_float(tf.not_equal(labels, 0))
      xentropy = xentropy * weights
      loss = tf.reduce_sum(xentropy * weights) / tf.reduce_sum(weights)

      return loss


class BasicSampledSequenceLoss(Loss):
  """
  Basic sequence-to-sequence loss. This one does not use one-hot encodings
  """
  @staticmethod
  def get_required_params():
    return dict(Loss.get_required_params(), **{
        'tgt_vocab_size': int,
        'batch_size': int,
    })

  @staticmethod
  def get_optional_params():
    return dict(Loss.get_optional_params(), **{
        'offset_target_by_one': bool,
        'average_across_timestep': bool,
        'do_mask': bool,
        'hid_dim': int,
    })

  def __init__(self, params, model, name="basic_sampled_sequence_loss"):
    """Constructor.

    Args:
      params (dict): dictionary with loss parameters.
        Should contain the following:
        * tgt_vocab_size: Target vocabulary size
        * batch_size_per_gpu: Size of the per-worker batch
        * offset_target_by_one: (default: True). Keep it true for
        auto-regressive models
        * average_across_timestep: (default: False). If True, will average
          loss across timesteps, else it will sum across timesteps
        * do_mask: (default: True) whether to mask based on tgt_lengths
          (which is passed as part of loss_input_dict to compute_loss
          and has to be not None then)
    """
    super(BasicSampledSequenceLoss, self).__init__(params, model, name)
    self._tgt_vocab_size = self.params["tgt_vocab_size"]
    self._batch_size = self.params["batch_size"]
    self._offset_target_by_one = self.params.get("offset_target_by_one", True)
    self._average_across_timestep = self.params.get("average_across_timestep", False)
    self._do_mask = self.params.get("do_mask", True)

  def _compute_loss(self, input_dict):
    """Computes cross entropy based sequence-to-sequence loss.

    Args:
      input_dict (dict): inputs to compute loss::
        {
              "logits": logits tensor of shape [batch_size, T, dim]
              "target_sequence": tensor of shape [batch_size, T]
              "tgt_lengths": tensor of shape [batch_size] or None
        }

    Returns:
       Singleton loss tensor
    """
    target_sequence = input_dict['target_tensors'][0]
    tgt_lengths = input_dict['target_tensors'][1]

    if 'weights' in input_dict['decoder_output']:
      print('DOING SAMPLED LOSS')
      inputs = input_dict["decoder_output"]['inputs']
      self._hid_dim = inputs.get_shape().as_list()[-1]
      inputs = tf.reshape(inputs, (-1, self._hid_dim))
      targets = tf.reshape(target_sequence, (-1, 1))
      crossent = tf.nn.sampled_softmax_loss(input_dict["decoder_output"]['weights'], 
                                            input_dict["decoder_output"]['bias'], 
                                            targets, 
                                            inputs,
                                            input_dict['decoder_output']['num_sampled'],
                                            self._tgt_vocab_size)
      if self._average_across_timestep:
        loss = tf.reduce_mean(crossent)
      else:
        loss = tf.reduce_sum(crossent)
        loss /= self._batch_size

    else:
      logits = input_dict["decoder_output"]["logits"]

      if self._offset_target_by_one:
        # this is necessary for auto-regressive models
        current_ts = tf.to_int32(tf.minimum(
            tf.shape(target_sequence)[1],
            tf.shape(logits)[1],
        )) - 1

        logits = tf.slice(
            logits,
            begin=[0, 0, 0],
            size=[-1, current_ts, -1],
        )                                 
        target_sequence = tf.slice(target_sequence,
                                   begin=[0, 1],
                                   size=[-1, current_ts])
      else:
        current_ts = tf.to_int32(tf.minimum(
            tf.shape(target_sequence)[1],
            tf.shape(logits)[1],
        ))

      # Cast logits after potential slice
      if logits.dtype.base_dtype != tf.float32:
        logits = tf.cast(logits, tf.float32)

      if self._do_mask:
        if tgt_lengths is None:
          raise ValueError("If you are masking loss, tgt_lengths can't be None")
        mask = tf.sequence_mask(lengths=tgt_lengths - 1,
                                maxlen=current_ts,
                                dtype=logits.dtype)
      else:
        mask = tf.cast(tf.ones_like(target_sequence), logits.dtype)

      crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tf.reshape(target_sequence, shape=[-1]),
          logits=tf.reshape(logits, shape=[-1, self._tgt_vocab_size]),
      )


      if self._average_across_timestep:
        loss = tf.reduce_mean(crossent * tf.reshape(mask, shape=[-1]))
      else:
        loss = tf.reduce_sum(crossent * tf.reshape(mask, shape=[-1]))
        loss /= self._batch_size
    return loss
   