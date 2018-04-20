# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
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
    })

  @staticmethod
  def get_optional_params():
    return dict(Loss.get_optional_params(), **{
      'offset_target_by_one': bool,
      'average_across_timestep': bool,
      'do_mask': bool,
    })

  def __init__(self, params, name="basic_sequence_loss"):
    """
    Constructor
    :param params - dictionary with loss parameters.
    Should contain the following:
    * tgt_vocab_size: Target vocabulary size
    * batch_size_per_gpu: Size of the per-worker batch
    * offset_target_by_one: (default: True). Keep it true for
    auto-regressive models
    * average_across_timestep: (default: False). If True, will average
    loss across timesteps, else it will sum across timesteps
    * do_mask: (default: True) whether to mask based on tgt_lengths
    (which is passed as part of loss_input_dict to compute_loss and has to be
    not None then)
    """
    super(BasicSequenceLoss, self).__init__(params, name)
    self._tgt_vocab_size = self.params["tgt_vocab_size"]
    self._batch_size_per_gpu = self.params["batch_size_per_gpu"]
    self._offset_target_by_one = self.params.get(
      "offset_target_by_one", True)
    self._average_across_timestep = self.params.get(
      "average_across_timestep", False)
    self._do_mask = self.params.get("do_mask", True)

  def _compute_loss(self, input_dict):
    """
    Computes cross entropy based sequence-to-sequence loss
    :param input_dict: inputs to compute loss
    {
          "logits": logits tensor of shape [batch_size, T, dim]
          "target_sequence": tensor of shape [batch_size, T]
          "tgt_lengths": tensor of shape [batch_size] or None
    }
    :return: Singleton loss tensor
    """
    logits = input_dict["logits"]
    target_sequence = input_dict["target_sequence"]
    tgt_lengths = input_dict["tgt_lengths"]

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
                              dtype=logits.dtype)  # TODO: why store in float?
    else:
      mask = tf.cast(tf.ones_like(target_sequence), logits.dtype)

    """
    if self._average_across_timestep:
      loss = tf.contrib.seq2seq.sequence_loss(
        logits=logits,
        targets=target_sequence,
        weights=mask,
        average_across_timesteps=self._average_across_timestep,
        average_across_batch=True,
        softmax_loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits,
      )
    else:
      crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.reshape(target_sequence, shape=[-1]),
        logits=tf.reshape(logits, shape=[-1, self._tgt_vocab_size]),
      )
      loss = tf.reduce_sum(crossent * tf.reshape(mask, shape=[-1]))      
      loss /= self._batch_size_per_gpu
    """
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(target_sequence, shape=[-1]),
      logits=tf.reshape(logits, shape=[-1, self._tgt_vocab_size]),
    )
    if self._average_across_timestep:
      loss = tf.reduce_mean(crossent * tf.reshape(mask, shape=[-1]))
    else:
      loss = tf.reduce_sum(crossent * tf.reshape(mask, shape=[-1]))
      loss /= self._batch_size_per_gpu
    return loss


class CrossEntropyWithSmoothing(Loss):
  """
  Softmax cross entropy loss with label smoothing. This one uses one-hot encodings
  for labels
  """
  @staticmethod
  def get_required_params():
    return dict(Loss.get_required_params(), **{
      'tgt_vocab_size': int,
    })

  @staticmethod
  def get_optional_params():
    return dict(Loss.get_optional_params(), **{
      'offset_target_by_one': bool,
      'average_across_timestep': bool,
      'do_mask': bool,
      'label_smoothing': float,
    })

  def __init__(self, params, name="cross_entropy_with_smoothing"):
    """
    Constructor
    :param params - dictionary with loss parameters.
    Should contain the following:
    * tgt_vocab_size: Target vocabulary size
    * batch_size_per_gpu: Size of the per-worker batch
    * offset_target_by_one: (default: True). Keep it true for
    auto-regressive models    
    * do_mask: (default: True) whether to mask based on tgt_lengths
    (which is passed as part of loss_input_dict to compute_loss and has to be
    not None then)
    """
    super(CrossEntropyWithSmoothing, self).__init__(params, name)
    self._tgt_vocab_size = self.params["tgt_vocab_size"]
    self._batch_size_per_gpu = self.params["batch_size_per_gpu"]
    self._offset_target_by_one = self.params.get(
      "offset_target_by_one", True)
    self._do_mask = self.params.get("do_mask", True)
    self._label_smoothing = self.params.get("label_smoothing", 0.0)
    self._average_across_timestep = self.params.get(
      "average_across_timestep", False)

  def _compute_loss(self, input_dict):
    """
    Computes cross entropy based sequence-to-sequence loss with label smoothing
    :param input_dict: inputs to compute loss
    {
          "logits": logits tensor of shape [batch_size, T, dim]
          "target_sequence": tensor of shape [batch_size, T]
          "tgt_lengths": tensor of shape [batch_size] or None
    }
    :return: Singleton loss tensor
    """
    logits = input_dict["logits"]
    target_sequence = input_dict["target_sequence"]
    tgt_lengths = input_dict["tgt_lengths"]

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

    labels = tf.one_hot(indices=tf.reshape(target_sequence, shape=[-1]), depth=self._tgt_vocab_size)
    logits = tf.reshape(logits, shape=[-1, self._tgt_vocab_size])
    mask = tf.reshape(mask, shape=[-1])

    loss = tf.losses.softmax_cross_entropy(
              onehot_labels=labels,
              logits=logits,
              weights=mask,
              label_smoothing=self._label_smoothing,
              reduction=tf.losses.Reduction.NONE)

    loss = tf.reduce_sum(loss * tf.reshape(mask, shape=[-1]))
    if self._average_across_timestep:
      loss /= tf.reduce_sum(mask)
    else:
      loss /= self._batch_size_per_gpu
    return loss
