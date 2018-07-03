# Copyright (c) 2018 NVIDIA Corporation
"""
Losses to be used in seq2seq models
"""
from .cross_entropy_loss import CrossEntropyLoss
from .ctc_loss import CTCLoss
from .sequence_loss import BasicSequenceLoss, CrossEntropyWithSmoothing, \
                           PaddedCrossEntropyLossWithSmoothing
