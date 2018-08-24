# Copyright (c) 2018 NVIDIA Corporation
"""
This package contains various encoders.
An encoder typically takes data and produces representation.
"""
from .encoder import Encoder
from .rnn_encoders import UnidirectionalRNNEncoderWithEmbedding, \
                          BidirectionalRNNEncoderWithEmbedding, \
                          GNMTLikeEncoderWithEmbedding,\
                          GNMTLikeEncoderWithEmbedding_cuDNN
from .transformer_encoder import TransformerEncoder
from .ds2_encoder import DeepSpeech2Encoder
from .resnet_encoder import ResNetEncoder
from .tacotron2_encoder import Tacotron2Encoder
from .w2l_encoder import Wave2LetterEncoder
from .convs2s_encoder import ConvS2SEncoder
from .lm_encoders import LMEncoder
