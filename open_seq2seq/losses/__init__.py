# Copyright (c) 2018 NVIDIA Corporation
"""
Losses to be used in seq2seq models
"""
from .sequence_loss import BasicSequenceLoss, CrossEntropyWithSmoothing, \
  PaddedCrossEntropyLossWithSmoothing
from .ctc_loss import CTCLoss
from .cross_entropy_loss import CrossEntropyLoss
from .mean_squared_error_loss import MeanSquaredErrorLoss, BasicMeanSquaredErrorLoss, TacotronLoss