# Copyright (c) 2018 NVIDIA Corporation
"""
This package contains various decoder.
A Decoder typically takes representation and produces data.
"""
from .decoder import Decoder
from .fc_decoders import FullyConnectedCTCDecoder, FullyConnectedDecoder, FullyConnectedSCDecoder
from .rnn_decoders import RNNDecoderWithAttention, \
    BeamSearchRNNDecoderWithAttention
from .transformer_decoder import TransformerDecoder
from .convs2s_decoder import ConvS2SDecoder
from .lm_decoders import FakeDecoder
from .tacotron2_decoder import Tacotron2Decoder
from .las_decoder import ListenAttendSpellDecoder
from .jca_decoder import JointCTCAttentionDecoder
from .centaur_decoder import CentaurDecoder