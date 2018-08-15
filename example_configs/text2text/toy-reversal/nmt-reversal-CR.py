# pylint: skip-file
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from open_seq2seq.models import Text2Text
from open_seq2seq.decoders import RNNDecoderWithAttention, BeamSearchRNNDecoderWithAttention
from open_seq2seq.encoders import ConvS2SEncoder

from open_seq2seq.data.text2text.text2text import ParallelTextDataLayer
from open_seq2seq.losses import BasicSequenceLoss

from open_seq2seq.data.text2text.text2text import SpecialTextTokens
from open_seq2seq.optimizers.lr_policies import fixed_lr

"""
This configuration file describes convolutional encoder and rnn decoder with attention
on the toy task of reversing sequences
"""

base_model = Text2Text
d_model = 128
num_layers = 2

base_params = {
  "use_horovod": False,
  "num_gpus": 1,
  "batch_size_per_gpu": 64,
  "max_steps": 1000,
  "save_summaries_steps": 10,
  "print_loss_steps": 10,
  "print_samples_steps": 20,
  "eval_steps": 50,
  "save_checkpoint_steps": 200,

  "logdir": "ReversalTask-Conv-RNN",

  "optimizer": "Adam",
  "optimizer_params": {"epsilon": 1e-9},
  "lr_policy": fixed_lr,
  "lr_policy_params": {
    'learning_rate': 1e-3
  },

  "max_grad_norm": 3.0,
  "dtype": tf.float32,
  # "dtype": "mixed",
  # "loss_scaling": "Backoff",

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],

  "encoder": ConvS2SEncoder,
  "encoder_params": {
    "encoder_layers": num_layers,

    "src_emb_size": d_model,
    "att_layer_num": num_layers,
    "embedding_dropout_keep_prob": 0.9,
    "pad_embeddings_2_eight": True,

    "hidden_dropout_keep_prob": 0.9,

    "conv_nchannels_kwidth": [(d_model, 3)] * num_layers,

    "max_input_length": 100,

    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,
  },

  "decoder": RNNDecoderWithAttention,
  "decoder_params": {
    "core_cell": tf.nn.rnn_cell.LSTMCell,
    "core_cell_params": {
      "num_units": d_model,
    },
    "decoder_layers": num_layers,

    "decoder_dp_input_keep_prob": 0.8,
    "decoder_dp_output_keep_prob": 1.0,
    "decoder_use_skip_connections": False,

    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,

    "tgt_emb_size": d_model,
    "attention_type": "luong",
    "luong_scale": False,
    "attention_layer_size": 128,
  },

  "loss": BasicSequenceLoss,
  "loss_params": {
    "offset_target_by_one": True,
    "average_across_timestep": True,
    "do_mask": True
  }
}

train_params = {
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "src_vocab_file": "toy_text_data/vocab/source.txt",
    "tgt_vocab_file": "toy_text_data/vocab/target.txt",
    "source_file": "toy_text_data/train/source.txt",
    "target_file": "toy_text_data/train/target.txt",
    "shuffle": True,
    "repeat": True,
    "max_length": 56,
    "delimiter": " ",
    "special_tokens_already_in_vocab": False,
  },
}

eval_params = {
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "src_vocab_file": "toy_text_data/vocab/source.txt",
    "tgt_vocab_file": "toy_text_data/vocab/target.txt",
    "source_file": "toy_text_data/dev/source.txt",
    "target_file": "toy_text_data/dev/target.txt",
    "shuffle": False,
    "repeat": True,
    "max_length": 56,
    "delimiter": " ",
    "special_tokens_already_in_vocab": False,
  },
}

infer_params = {
  "batch_size_per_gpu": 1,
  "decoder": BeamSearchRNNDecoderWithAttention,
  "decoder_params": {
    "core_cell": tf.nn.rnn_cell.LSTMCell,
    "core_cell_params": {
      "num_units": d_model,
    },
    "decoder_layers": num_layers,
    "decoder_dp_input_keep_prob": 0.8,
    "decoder_dp_output_keep_prob": 1.0,
    "decoder_use_skip_connections": False,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,
    "tgt_emb_size": d_model,
    "attention_type": "luong",
    "luong_scale": False,
    "attention_layer_size": d_model,
    "beam_width": 5,
    "length_penalty": 1.0,
  },

  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "src_vocab_file": "toy_text_data/vocab/source.txt",
    "tgt_vocab_file": "toy_text_data/vocab/source.txt",
    "source_file": "toy_text_data/test/source.txt",
    "target_file": "toy_text_data/test/source.txt",
    "shuffle": False,
    "repeat": False,
    "max_length": 256,
    "delimiter": " ",
    "special_tokens_already_in_vocab": False,
  },
}
