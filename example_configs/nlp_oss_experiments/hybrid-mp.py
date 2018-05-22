from __future__ import absolute_import, division, print_function
import tensorflow as tf

from open_seq2seq.models import Text2Text
from open_seq2seq.encoders import TransformerEncoder
from open_seq2seq.decoders import RNNDecoderWithAttention, \
  BeamSearchRNNDecoderWithAttention
from open_seq2seq.data.text2text.text2text import ParallelTextDataLayer
from open_seq2seq.losses import BasicSequenceLoss
from open_seq2seq.data.text2text.text2text import SpecialTextTokens
from open_seq2seq.optimizers.lr_policies import transformer_policy

data_root = "/data/wmt16_s2s/"

base_model = Text2Text

base_params = {
  "use_horovod": False,
  "num_gpus": 8,
  "max_steps": 340000,
  "batch_size_per_gpu": 16,
  "save_summaries_steps": 50,
  "print_loss_steps": 48,
  "print_samples_steps": 48,
  "eval_steps": 1000,
  "save_checkpoint_steps": 2001,
  "logdir": "Hybrid-MP-luong10-P8-AAT",
  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": transformer_policy,
  "lr_policy_params": {
    "learning_rate": 2.0,
    "warmup_steps": 16000,
    "d_model": 1024,
  },
  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],
  #"dtype": tf.float32,
  "dtype": "mixed",
  "automatic_loss_scaling": "Backoff",
  "encoder": TransformerEncoder,
  "encoder_params": {
    "encoder_layers": 6,
    "hidden_size": 1024,
    "num_heads": 8,
    "attention_dropout": 0.2,
    "filter_size": 4*512,
    "relu_dropout": 0.2,
    "layer_postprocess_dropout": 0.2,
  },

  "decoder": RNNDecoderWithAttention,
  "decoder_params": {
    "initializer": tf.random_uniform_initializer,
    "initializer_params": {
       "minval": -0.1,
       "maxval": 0.1,
     },
    "decoder_cell_type": "lstm",
    "decoder_cell_units": 1024,
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
    "pad_vocab_to_eight": True,
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
    "decoder_cell_type": "lstm",
    "decoder_cell_units": 1024,
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
