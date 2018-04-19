# Copyright (c) 2018 NVIDIA Corporation
"""
Losses to be used in seq2seq models
"""
from .sequence_loss import BasicSequenceLoss, CrossEntropyWithSmoothing
from .ctc_loss import CTCLoss
