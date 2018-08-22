# pylint: skip-file
from __future__ import absolute_import, division, print_function
from open_seq2seq.models import Text2Text
from open_seq2seq.encoders import TransformerEncoder
from open_seq2seq.decoders import TransformerDecoder
from open_seq2seq.data.text2text.text2text import ParallelTextDataLayer
from open_seq2seq.losses import PaddedCrossEntropyLossWithSmoothing
from open_seq2seq.data.text2text.text2text import SpecialTextTokens
from open_seq2seq.optimizers.lr_policies import transformer_policy
import tensorflow as tf

"""
This configuration file describes a tiny variant of Transformer model from
https://arxiv.org/abs/1706.03762 on the toy task of reversing sequences
"""

base_model = Text2Text
d_model = 128
num_layers = 2

base_params = {
  "use_horovod": False,
  "num_gpus": 1,
  "batch_size_per_gpu": 64,
  "max_steps": 800,
  "save_summaries_steps": 50,
  "print_loss_steps": 50,
  "print_samples_steps": 50,
  "eval_steps": 50,
  "save_checkpoint_steps": 300,
  "logdir": "ReversalTask-Transformer-Transformer",
  "dtype": tf.float32,
  # "dtype": "mixed",
  # "loss_scaling": "Backoff",

  "optimizer": tf.contrib.opt.LazyAdamOptimizer,
  "optimizer_params": {
    "beta1": 0.9,
    "beta2": 0.997,
    "epsilon": 0.000000001,
  },
  "lr_policy": transformer_policy,
  "lr_policy_params": {
    "learning_rate": 1.0,
    "warmup_steps": 200,
    "d_model": d_model,
  },
  "encoder": TransformerEncoder,
  "encoder_params": {
    "encoder_layers": num_layers,
    "hidden_size": d_model,
    "num_heads": 8,
    "attention_dropout": 0.1,
    "filter_size": 4*d_model,
    "relu_dropout": 0.1,
    "layer_postprocess_dropout": 0.1,
  },

  "decoder": TransformerDecoder,
  "decoder_params": {
    "layer_postprocess_dropout": 0.1,
    "num_hidden_layers": num_layers,
    "hidden_size": d_model,
    "num_heads": 8,
    "attention_dropout": 0.1,
    "relu_dropout": 0.1,
    "filter_size": 4*d_model,
    "beam_size": 5,
    "alpha": 1.0,
    "extra_decode_length": 2,
    "EOS_ID": SpecialTextTokens.EOS_ID.value,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,
  },

  "loss": PaddedCrossEntropyLossWithSmoothing,
  "loss_params": {}
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
    "use_start_token": False,
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
    "use_start_token": False,
  },
}

infer_params = {
  "batch_size_per_gpu": 1,
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
    "use_start_token": False,
  },
}
