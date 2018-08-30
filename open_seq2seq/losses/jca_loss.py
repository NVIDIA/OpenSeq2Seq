# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from .loss import Loss
from .ctc_loss import CTCLoss
from .sequence_loss import BasicSequenceLoss

# To-Do Replace this with a generic multi-task loss.


class MultiTaskCTCEntropyLoss(Loss):
  """
  MultiTask CTC and cross entropy loss.
  """
  @staticmethod
  def get_required_params():
    return dict(Loss.get_required_params(), **{
        'ctc_loss_params': dict,
        'seq_loss_params': dict,
        'lambda_value': float,
        'tgt_vocab_size': int,
        'batch_size': int,
    })

  @staticmethod
  def get_optional_params():
    return dict(Loss.get_optional_params(), **{
    })

  def __init__(self, params, model, name="basic_sequence_loss"):
    """Constructor.

    Args:
      params (dict): dictionary with loss parameters.
        Should contain the following:
        * ctc_loss_params: Parameters required for CTC loss.
        * seq_loss_params: Parameters required for Sequence loss.
        * lambda_value: lambda value used to combine the two losses.
        * tgt_vocab_size: Target vocabulary size.
        * batch_size: Size of the per-worker batch.
    """
    super(MultiTaskCTCEntropyLoss, self).__init__(params, model, name)
    self.ctc_loss_params = self.params["ctc_loss_params"]
    self.seq_loss_params = self.params["seq_loss_params"]
    self.lambda_value = self.params["lambda_value"]

    self.seq_loss_params["batch_size"] = self.params["batch_size"]
    self.seq_loss_params["tgt_vocab_size"] = self.params["tgt_vocab_size"]

    self.ctc_loss = CTCLoss(self.ctc_loss_params, model)
    self.seq_loss = BasicSequenceLoss(self.seq_loss_params, model)

  def _compute_loss(self, input_dict):
    """Computes multi-task ctc and cross entropy loss.

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

    ctc_loss_input_dict = {
        "decoder_output": input_dict['decoder_output']['ctc_outputs'],
        "target_tensors": input_dict['target_tensors'],
    }

    seq_loss_input_dict = {
        "decoder_output": input_dict['decoder_output']['seq_outputs'],
        "target_tensors": input_dict['target_tensors'],
    }

    ctc_loss_value = self.ctc_loss.compute_loss(ctc_loss_input_dict)
    sequence_loss_value = self.seq_loss.compute_loss(seq_loss_input_dict)

    return (1 - self.lambda_value) * sequence_loss_value + self.lambda_value * ctc_loss_value
