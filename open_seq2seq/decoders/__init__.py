# Copyright (c) 2018 NVIDIA Corporation
"""
This package contains various decoder.
A Decoder typically takes representation and produces data.
"""
from .decoder import Decoder
from .rnn_decoders import RNNDecoderWithAttention, \
                          BeamSearchRNNDecoderWithAttention
from .transformer_decoders import TransformerDecoder
from .fc_decoder import FullyConnectedTimeDecoder, FullyConnectedCTCDecoder
