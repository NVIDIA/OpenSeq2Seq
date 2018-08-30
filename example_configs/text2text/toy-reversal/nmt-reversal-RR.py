# pylint: skip-file
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from open_seq2seq.models import Text2Text
from open_seq2seq.encoders import BidirectionalRNNEncoderWithEmbedding
from open_seq2seq.decoders import RNNDecoderWithAttention, \
  BeamSearchRNNDecoderWithAttention
from open_seq2seq.data.text2text.text2text import ParallelTextDataLayer
from open_seq2seq.losses import BasicSequenceLoss
from open_seq2seq.data.text2text.text2text import SpecialTextTokens
from open_seq2seq.optimizers.lr_policies import fixed_lr

"""
This configuration file describes classic RNN-based encoder-decoder model
with attention on the toy task of reversing sequences
"""

base_model = Text2Text

base_params = {
  "use_horovod": False,
  #"iter_size": 10,
  # set this to number of available GPUs
  "num_gpus": 1,
  "batch_size_per_gpu": 64,
  "max_steps": 800,
  "save_summaries_steps": 10,
  "print_loss_steps": 10,
  "print_samples_steps": 20,
  "eval_steps": 50,
  "save_checkpoint_steps": 300,
  "logdir": "ReversalTask-RNN-RNN",

  "optimizer": "Adam",
  "optimizer_params": {"epsilon": 1e-4},
  "lr_policy": fixed_lr,
  "lr_policy_params": {
    'learning_rate': 0.001
  },
  "max_grad_norm": 3.0,
  "dtype": tf.float32,
  # "dtype": "mixed",
  # "loss_scaling": "Backoff",

  "encoder": BidirectionalRNNEncoderWithEmbedding,
  "encoder_params": {
    "core_cell": tf.nn.rnn_cell.LSTMCell,
    "core_cell_params": {
      "num_units": 128,
      "forget_bias": 1.0,
    },
    "encoder_layers": 1,
    "encoder_dp_input_keep_prob": 0.8,
    "encoder_dp_output_keep_prob": 1.0,
    "encoder_use_skip_connections": False,
    "src_emb_size": 128,
  },

  "decoder": RNNDecoderWithAttention,
  "decoder_params": {
    "core_cell": tf.nn.rnn_cell.LSTMCell,
    "core_cell_params": {
      "num_units": 128,
      # "forget_bias": 1.0,
    },
    "decoder_layers": 1,
    "decoder_dp_input_keep_prob": 0.8,
    "decoder_dp_output_keep_prob": 1.0,
    "decoder_use_skip_connections": False,    
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "tgt_emb_size": 128,
    "attention_type": "luong",
    "luong_scale": False,
    "attention_layer_size": 128,
  },

  "loss": BasicSequenceLoss,
  "loss_params": {    
    "offset_target_by_one": True,
    "average_across_timestep": False,
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
    # because we evaluate many times
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
    #"decoder_cell_type": "lstm",
    #"decoder_cell_units": 128,
    "core_cell": tf.nn.rnn_cell.LSTMCell,
    "core_cell_params": {
      "num_units": 128,
      "forget_bias": 1.0,
    },
    "decoder_layers": 1,
    "decoder_dp_input_keep_prob": 0.8,
    "decoder_dp_output_keep_prob": 1.0,
    "decoder_use_skip_connections": False,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,
    "tgt_emb_size": 128,
    "attention_type": "luong",
    "luong_scale": False,
    "attention_layer_size": 128,
    "beam_width": 10,
    "length_penalty": 1.0,
  },

  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "src_vocab_file": "toy_text_data/vocab/source.txt",
    "tgt_vocab_file": "toy_text_data/vocab/source.txt",
    "source_file": "toy_text_data/test/source.txt",
    "target_file": "toy_text_data/test/target.txt",
    "shuffle": False,
    "repeat": False,
    "max_length": 256,
    "delimiter": " ",
    "special_tokens_already_in_vocab": False,
  },
}