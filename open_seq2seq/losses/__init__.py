# Copyright (c) 2018 NVIDIA Corporation
"""
Losses to be used in seq2seq models
"""
from .sequence_loss import BasicSequenceLoss, CrossEntropyWithSmoothing, \
  PaddedCrossEntropyLossWithSmoothing, BasicSampledSequenceLoss
from .ctc_loss import CTCLoss
from .cross_entropy_loss import CrossEntropyLoss
from .tacotron_loss import TacotronLoss