# Copyright (c) 2018 NVIDIA Corporation
"""
This package contains various encoders.
An encoder typically takes data and produces representation.
"""
from .encoder import Encoder
from .rnn_encoders import UnidirectionalRNNEncoderWithEmbedding, \
                          BidirectionalRNNEncoderWithEmbedding, \
                          GNMTLikeEncoderWithEmbedding
from .transformer_encoders import TransformerEncoder
from .ds2_encoder import DeepSpeech2Encoder
