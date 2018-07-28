# Copyright (c) 2018 NVIDIA Corporation
"""
This package contains various decoder.
A Decoder typically takes representation and produces data.
"""
from .convs2s_decoder import ConvS2SDecoder
from .decoder import Decoder
from .fc_decoders import FullyConnectedCTCDecoder, FullyConnectedDecoder
from .rnn_decoders import RNNDecoderWithAttention, \
                          BeamSearchRNNDecoderWithAttention
from .transformer_decoder import TransformerDecoder
from .tacotron2_decoder import Tacotron2Decoder
from .convs2s_decoder import ConvS2SDecoder

from .convs2s_decoder2 import ConvS2SDecoder2