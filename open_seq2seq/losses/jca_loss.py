# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from .loss import Loss
from .ctc_loss import CTCLoss
from .sequence_loss import BasicSequenceLoss

class MultiTaskCTCEntropyLoss(Loss):
  """
  Basic sequence-to-sequence loss. This one does not use one-hot encodings
  """
  @staticmethod
  def get_required_params():
    return dict(Loss.get_required_params(), **{
        'ctc_loss_params': dict,
        'seq_loss_params': dict,
        'lambda_value': float,
        'lambda_params': dict,
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
    """
    super(MultiTaskCTCEntropyLoss, self).__init__(params, model, name)
    self.ctc_loss_params = self.params["ctc_loss_params"]
    self.seq_loss_params = self.params["seq_loss_params"]
    self.lambda_value = self.params["lambda_value"] 
    self.lambda_params = self.params["lambda_params"] 

    self.seq_loss_params["batch_size"] = self.params["batch_size"]
    self.seq_loss_params["tgt_vocab_size"] = self.params["tgt_vocab_size"]


    self.ctc_loss = CTCLoss(self.ctc_loss_params, model)
    self.seq_loss = BasicSequenceLoss(self.seq_loss_params, model)

  def _compute_loss(self, input_dict):
    """Computes multi-task ctc and cross entropy loss.
    """

    ctc_loss_input_dict = {
              "decoder_output": input_dict['decoder_output']['ctc_outputs'],
              "target_tensors": input_dict['target_tensors'],
    }

    seq_loss_input_dict = {
              "decoder_output": input_dict['decoder_output']['las_outputs'],
              "target_tensors": input_dict['target_tensors'],
    }

    ctc_loss_value = self.ctc_loss.compute_loss(ctc_loss_input_dict)
    sequence_loss_value = self.seq_loss.compute_loss(seq_loss_input_dict)

    global_step = tf.train.get_or_create_global_step()
    values = self.lambda_params["values"]
    boundaries = self.lambda_params["boundaries"]
    #self.lambda_value = tf.train.piecewise_constant(global_step, boundaries, values)

    return (1-self.lambda_value)*sequence_loss_value + self.lambda_value*ctc_loss_value