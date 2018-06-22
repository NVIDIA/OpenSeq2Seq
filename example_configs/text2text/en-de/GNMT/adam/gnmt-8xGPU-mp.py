from __future__ import absolute_import, division, print_function
import tensorflow as tf

from open_seq2seq.models import Text2Text
from open_seq2seq.encoders import GNMTLikeEncoderWithEmbedding_cuDNN
from open_seq2seq.decoders import RNNDecoderWithAttention, \
  BeamSearchRNNDecoderWithAttention
from open_seq2seq.data.text2text.text2text import ParallelTextDataLayer
from open_seq2seq.losses import BasicSequenceLoss
from open_seq2seq.data.text2text.text2text import SpecialTextTokens
from open_seq2seq.optimizers.lr_policies import exp_decay

data_root = "/data/wmt16_s2s/"

base_model = Text2Text
pad_vocabs_2_eight = True

base_params = {
  "use_horovod": True,
  "num_gpus": 1,
  "max_steps": 42500,
  "batch_size_per_gpu": 64,
  "save_summaries_steps": 100,
  "print_loss_steps": 101,
  "print_samples_steps": 101,
  "eval_steps": 2000,
  "save_checkpoint_steps": 10625,
  "logdir": "GNMT-8xGPU-mp",
  "optimizer": "Adam",
  "optimizer_params": {},
  # luong10 decay scheme
  "lr_policy": exp_decay,
  "lr_policy_params": {
    "learning_rate": 0.0008,
    "begin_decay_at": 21250,
    "decay_steps": 2125,
    "decay_rate": 0.5,
    "use_staircase_decay": True,
    "min_lr": 0.0000005,
  },
  "max_grad_norm": 5.0,
  #"dtype": tf.float32,
  "dtype": "mixed",
  "loss_scaling": "Backoff",
  "encoder": GNMTLikeEncoderWithEmbedding_cuDNN,
  "encoder_params": {
    "initializer": tf.random_uniform_initializer,
    "initializer_params": {
      "minval": -0.1,
      "maxval": 0.1,
    },
    "encoder_cell_type": "lstm",
    "encoder_cell_units": 1024,
    "encoder_layers": 8,
    "encoder_dp_output_keep_prob": 1.0,
    "src_emb_size": 1024,
  },

  "decoder": RNNDecoderWithAttention,
  "decoder_params": {
    "initializer": tf.random_uniform_initializer,
    "initializer_params": {
       "minval": -0.1,
       "maxval": 0.1,
     },
    "core_cell": tf.nn.rnn_cell.LSTMCell,
    "core_cell_params": {
        "num_units": 1024,
        "forget_bias": 1.0,
    },

    "decoder_layers": 8,
    "decoder_dp_input_keep_prob": 0.8,
    "decoder_dp_output_keep_prob": 1.0,
    "decoder_use_skip_connections": True,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,

    "tgt_emb_size": 1024,
    "attention_type": "gnmt_v2",
    "attention_layer_size": 1024,
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
    "pad_vocab_to_eight": pad_vocabs_2_eight,
    "src_vocab_file": data_root+"vocab.bpe.32000",
    "tgt_vocab_file": data_root+"vocab.bpe.32000",
    "source_file": data_root+"train.tok.clean.bpe.32000.en",
    "target_file": data_root+"train.tok.clean.bpe.32000.de",
    "delimiter": " ",
    "shuffle": True,
    "repeat": True,
    "map_parallel_calls": 16,
    "prefetch_buffer_size": 8,
    "max_length": 50,
  },
}
eval_params = {
  "batch_size_per_gpu": 16,
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "pad_vocab_to_eight": pad_vocabs_2_eight,
    "src_vocab_file": data_root+"vocab.bpe.32000",
    "tgt_vocab_file": data_root+"vocab.bpe.32000",
    "source_file": data_root+"newstest2013.tok.bpe.32000.en",
    "target_file": data_root+"newstest2013.tok.bpe.32000.de",
    "delimiter": " ",
    "shuffle": False,
    "repeat": True,
    "map_parallel_calls": 16,
    "prefetch_buffer_size": 1,
    "max_length": 32,
  },
}

infer_params = {
  "batch_size_per_gpu": 1,
  "decoder": BeamSearchRNNDecoderWithAttention,
  "decoder_params": {
    "beam_width": 10,
    "length_penalty": 1.0,
    "core_cell": tf.nn.rnn_cell.LSTMCell,
    "core_cell_params": {
      "num_units": 1024,
    },
    "decoder_layers": 8,
    "decoder_dp_input_keep_prob": 0.8,
    "decoder_dp_output_keep_prob": 1.0,
    "decoder_use_skip_connections": True,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,
    "tgt_emb_size": 1024,
    "attention_type": "gnmt_v2",
    "attention_layer_size": 1024,
  },

  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "pad_vocab_to_eight": pad_vocabs_2_eight,
    "src_vocab_file": data_root+"vocab.bpe.32000",
    "tgt_vocab_file": data_root+"vocab.bpe.32000",
    "source_file": data_root+"newstest2014.tok.bpe.32000.en",
    # this is intentional
    "target_file": data_root+"newstest2014.tok.bpe.32000.en",
    "delimiter": " ",
    "shuffle": False,
    "repeat": False,
    "max_length": 512,
  },
}
