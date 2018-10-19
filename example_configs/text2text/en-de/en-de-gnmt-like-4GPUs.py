# pylint: skip-file
from __future__ import absolute_import, division, print_function
import tensorflow as tf

from open_seq2seq.models import Text2Text
from open_seq2seq.encoders import GNMTLikeEncoderWithEmbedding
from open_seq2seq.decoders import RNNDecoderWithAttention, \
  BeamSearchRNNDecoderWithAttention
from open_seq2seq.data.text2text.text2text import ParallelTextDataLayer
from open_seq2seq.losses import BasicSequenceLoss
from open_seq2seq.data.text2text.text2text import SpecialTextTokens
from open_seq2seq.optimizers.lr_policies import exp_decay

data_root = "[REPLACE THIS TO THE PATH WITH YOUR WMT DATA]"

base_model = Text2Text

base_params = {
  "use_horovod": False,
  "num_gpus": 4,
  "max_steps": 340000,
  "batch_size_per_gpu": 32,
  "save_summaries_steps": 50,
  "print_loss_steps": 48,
  "print_samples_steps": 48,
  "eval_steps": 4001,
  "save_checkpoint_steps": 4000,
  "logdir": "GNMT-4GPUs-FP32",
  "optimizer": "Adam",
  "optimizer_params": {},
  # luong10 decay scheme
  "lr_policy": exp_decay,
  "lr_policy_params": {
    "learning_rate": 0.0008,
    "begin_decay_at": 170000,
    "decay_steps": 17000,
    "decay_rate": 0.5,
    "use_staircase_decay": True,
    "min_lr": 0.0000005,
  },
  #"summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
  #              'variable_norm', 'gradient_norm', 'global_gradient_norm'],
  "max_grad_norm": 32768.0,
  "dtype": tf.float32, # to enable mixed precision, comment this line and uncomment two below lines
  #"dtype": "mixed",
  #"loss_scaling": "Backoff",
  "encoder": GNMTLikeEncoderWithEmbedding,
  "encoder_params": {
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
    "encoder_layers": 7,
    "encoder_dp_input_keep_prob": 0.8,
    "encoder_dp_output_keep_prob": 1.0,
    "encoder_use_skip_connections": True,
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
    "pad_vocab_to_eight": True,
    "src_vocab_file": data_root + "m_common.vocab",
    "tgt_vocab_file": data_root + "m_common.vocab",
    "source_file": data_root + "train.clean.en.shuffled.BPE_common.32K.tok",
    "target_file": data_root + "train.clean.de.shuffled.BPE_common.32K.tok",
    "delimiter": " ",
    "shuffle": True,
    "shuffle_buffer_size": 25000,
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
    "pad_vocab_to_eight": True,
    "src_vocab_file": data_root+"m_common.vocab",
    "tgt_vocab_file": data_root+"m_common.vocab",
    "source_file": data_root+"wmt13-en-de.src.BPE_common.32K.tok",
    "target_file": data_root+"wmt13-en-de.ref.BPE_common.32K.tok",
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
        "forget_bias": 1.0,
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
    "pad_vocab_to_eight": True,
    "src_vocab_file": data_root+"m_common.vocab",
    "tgt_vocab_file": data_root+"m_common.vocab",
    "source_file": data_root+"wmt14-en-de.src.BPE_common.32K.tok",
    "target_file": data_root+"wmt14-en-de.src.BPE_common.32K.tok",
    "delimiter": " ",
    "shuffle": False,
    "repeat": False,
    "max_length": 512,
  },
}
