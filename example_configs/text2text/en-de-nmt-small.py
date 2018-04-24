from __future__ import absolute_import, division, print_function
import tensorflow as tf

from open_seq2seq.models import BasicText2TextWithAttention
from open_seq2seq.encoders import BidirectionalRNNEncoderWithEmbedding
from open_seq2seq.decoders import RNNDecoderWithAttention, \
  BeamSearchRNNDecoderWithAttention
from open_seq2seq.data.text2text import ParallelTextDataLayer
from open_seq2seq.losses import BasicSequenceLoss
from open_seq2seq.data.text2text import SpecialTextTokens

data_root = "[REPLACE THIS TO THE PATH WITH YOUR WMT DATA]"

# This model should run fine on single GPU such as 1080ti or better

base_params = {
  "use_horovod": False,
  "num_gpus": 1,
  "max_steps": 160082,
  "batch_size_per_gpu": 128,
  "save_summaries_steps": 50,
  "print_loss_steps": 48,
  "print_samples_steps": 48,
  "eval_steps": 1000,
  "save_checkpoint_steps": 2001,
  "logdir": "nmt-small-en-de",
  "base_model": BasicText2TextWithAttention,
  "model_params": {
    "optimizer": "Adam",
    "optimizer_params": {},
    "learning_rate": 0.001,
    "larc_mode": "clip",
    "larc_nu": 0.001,
    "dtype": tf.float32,
    #"dtype": "mixed",
    #"automatic_loss_scaling": "Backoff",
  },

  "encoder": BidirectionalRNNEncoderWithEmbedding,
  "encoder_params": {
    "initializer": tf.glorot_uniform_initializer,
    "encoder_cell_type": "lstm",
    "encoder_cell_units": 512,
    "encoder_layers": 2,
    "encoder_dp_input_keep_prob": 0.8,
    "encoder_dp_output_keep_prob": 1.0,
    "encoder_use_skip_connections": False,
    "src_emb_size": 512,
    "use_swap_memory": True,
  },

  "decoder": RNNDecoderWithAttention,
  "decoder_params": {
    "initializer": tf.glorot_uniform_initializer,
    "decoder_cell_type": "lstm",
    "decoder_cell_units": 512,
    "decoder_layers": 2,
    "decoder_dp_input_keep_prob": 0.8,
    "decoder_dp_output_keep_prob": 1.0,
    "decoder_use_skip_connections": False,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "tgt_emb_size": 512,
    "attention_type": "gnmt_v2",
    "attention_layer_size": 512,
    "use_swap_memory": True,
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
    "src_vocab_file": data_root+"vocab.bpe.32000",
    "tgt_vocab_file": data_root+"vocab.bpe.32000",
    "source_file": data_root+"train.tok.clean.bpe.32000.en",
    "target_file": data_root+"train.tok.clean.bpe.32000.de",
    "delimiter": " ",
    "shuffle": True,
    "repeat": True,
    "map_parallel_calls": 16,
    "prefetch_buffer_size": 2,
    "max_length": 50,
  },
}

eval_params = {
  "batch_size_per_gpu": 32,
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "src_vocab_file": data_root+"vocab.bpe.32000",
    "tgt_vocab_file": data_root+"vocab.bpe.32000",
    "source_file": data_root+"newstest2013.tok.bpe.32000.en",
    "target_file": data_root+"newstest2013.tok.bpe.32000.de",
    "delimiter": " ",
    "shuffle": False,
    "repeat": True,
    "map_parallel_calls": 16,
    "prefetch_buffer_size": 1,
    "max_length": 16,
  },
}

infer_params = {
  "batch_size_per_gpu": 1,
  "decoder": BeamSearchRNNDecoderWithAttention,
  "decoder_params": {
    "beam_width": 10,
    "length_penalty": 1.0,
    "decoder_cell_type": "lstm",
    "decoder_cell_units": 512,
    "decoder_layers": 2,
    "decoder_dp_input_keep_prob": 0.8,
    "decoder_dp_output_keep_prob": 1.0,
    "decoder_use_skip_connections": False,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,
    "tgt_emb_size": 512,
    "attention_type": "gnmt_v2",
    "attention_layer_size": 512,
  },
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "src_vocab_file": data_root+"vocab.bpe.32000",
    "tgt_vocab_file": data_root+"vocab.bpe.32000",
    "source_file": data_root+"newstest2014.tok.bpe.32000.en",
    # this is intentional
    "target_file": data_root+"newstest2014.tok.bpe.32000.en",
    "delimiter": " ",
    "shuffle": False,
    "repeat": False,
    "max_length": 256,
    "prefetch_buffer_size": 1,
  },
}
