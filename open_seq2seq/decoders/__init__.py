# Copyright (c) 2018 NVIDIA Corporation
"""
This package contains various decoder.
A Decoder typically takes representation and produces data.
"""
from .decoder import Decoder
from .rnn_decoders import RNNDecoderWithAttention, \
                          BeamSearchRNNDecoderWithAttention

from .transformer_decoder import TransformerDecoder
from .fc_decoders import FullyConnectedCTCDecoder, FullyConnectedDecoder

from .convs2s_decoder import ConvS2SDecoder
#from .convs2s_decoder_old import ConvS2SDecoder

