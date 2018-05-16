from __future__ import absolute_import, division, print_function
import tensorflow as tf
from open_seq2seq.models import BasicText2TextWithAttention
from open_seq2seq.encoders import TransformerEncoder
from open_seq2seq.decoders import RNNDecoderWithAttention, \
  BeamSearchRNNDecoderWithAttention
from open_seq2seq.data.text2text.text2text import ParallelTextDataLayer
from open_seq2seq.losses import BasicSequenceLoss
from open_seq2seq.data.text2text.text2text import SpecialTextTokens

"""
This configuration file describes classic RNN-based encoder-decoder model
with attention on the toy task of reversing sequences
"""

base_model = BasicText2TextWithAttention

base_params = {
  "use_horovod": False,
  # set this to number of available GPUs
  "num_gpus": 1,
  "batch_size_per_gpu": 64,
  "max_steps": 800,
  "save_summaries_steps": 10,
  "print_loss_steps": 10,
  "print_samples_steps": 20,
  "eval_steps": 50,
  "save_checkpoint_steps": 300,
  "logdir": "ReversalTask-Transformer-RNN",

  "optimizer": "Adam",
  "optimizer_params": {"epsilon": 1e-4},
  "learning_rate": 0.001,
  "max_grad_norm": 3.0,
  "dtype": tf.float32,

  "encoder": TransformerEncoder,
  "encoder_params": {
    "encoder_layers": 2,
    "hidden_size": 128,
    "num_heads": 8,
    "attention_dropout": 0.1,
    "filter_size": 4 * 128,
    "relu_dropout": 0.1,
    "layer_postprocess_dropout": 0.1,
  },

  "decoder": RNNDecoderWithAttention,
  "decoder_params": {
    "decoder_cell_type": "lstm",
    "decoder_cell_units": 128,
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
  },
}

infer_params = {
  "batch_size_per_gpu": 1,
  "decoder": BeamSearchRNNDecoderWithAttention,
  "decoder_params": {
    "decoder_cell_type": "lstm",
    "decoder_cell_units": 128,
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
  },
}