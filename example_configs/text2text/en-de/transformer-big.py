# pylint: skip-file
from __future__ import absolute_import, division, print_function
from open_seq2seq.models import Text2Text
from open_seq2seq.encoders import TransformerEncoder
from open_seq2seq.decoders import TransformerDecoder
from open_seq2seq.data.text2text.text2text import ParallelTextDataLayer
from open_seq2seq.losses import PaddedCrossEntropyLossWithSmoothing
from open_seq2seq.data.text2text.text2text import SpecialTextTokens
from open_seq2seq.data.text2text.tokenizer import EOS_ID
from open_seq2seq.optimizers.lr_policies import transformer_policy
import tensorflow as tf

"""
This configuration file describes a variant of Transformer model from
https://arxiv.org/abs/1706.03762
"""

base_model = Text2Text
d_model = 1024
num_layers = 6

# REPLACE THIS TO THE PATH WITH YOUR WMT DATA
data_root = "[REPLACE THIS TO THE PATH WITH YOUR WMT DATA]"

base_params = {
  "use_horovod": True,
  "num_gpus": 1, # when using Horovod we set number of workers with params to mpirun
  "batch_size_per_gpu": 256,  # this size is in sentence pairs, reduce it if you get OOM
  "max_steps": 300000,
  "save_summaries_steps": 100,
  "print_loss_steps": 100,
  "print_samples_steps": 100,
  "eval_steps": 4001,
  "save_checkpoint_steps": 299998,
  "logdir": "Transformer-BIG",
  #"dtype": tf.float32, # to enable mixed precision, comment this line and uncomment two below lines
  "dtype": "mixed",
  "loss_scaling": "Backoff",

  "optimizer": tf.contrib.opt.LazyAdamOptimizer,
  "optimizer_params": {
    "beta1": 0.9,
    "beta2": 0.997,
    "epsilon": 1e-09,
  },

  "lr_policy": transformer_policy,
  "lr_policy_params": {
    "learning_rate": 2.0,
    "warmup_steps": 8000,
    "d_model": d_model,
  },

  "encoder": TransformerEncoder,
  "encoder_params": {
    "encoder_layers": num_layers,
    "hidden_size": d_model,
    "num_heads": 16,
    "attention_dropout": 0.1,
    "filter_size": 4 * d_model,
    "relu_dropout": 0.3,
    "layer_postprocess_dropout": 0.3,
    "pad_embeddings_2_eight": True,
    "remove_padding": True,
  },

  "decoder": TransformerDecoder,
  "decoder_params": {
    "layer_postprocess_dropout": 0.3,
    "num_hidden_layers": num_layers,
    "hidden_size": d_model,
    "num_heads": 16,
    "attention_dropout": 0.1,
    "relu_dropout": 0.3,
    "filter_size": 4 * d_model,
    "beam_size": 4,
    "alpha": 0.6,
    "extra_decode_length": 50,
    "EOS_ID": EOS_ID,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,
  },

  "loss": PaddedCrossEntropyLossWithSmoothing,
  "loss_params": {
    "label_smoothing": 0.1,
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
    "max_length": 56,
  },
}

eval_params = {
  "batch_size_per_gpu": 16,
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "src_vocab_file": data_root+"m_common.vocab",
    "tgt_vocab_file": data_root+"m_common.vocab",
    "source_file": data_root+"wmt13-en-de.src.BPE_common.32K.tok",
    "target_file": data_root+"wmt13-en-de.ref.BPE_common.32K.tok",
    "delimiter": " ",
    "shuffle": False,
    "repeat": False,
    "max_length": 256,
    },
}

infer_params = {
  "batch_size_per_gpu": 1,
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "src_vocab_file": data_root+"m_common.vocab",
    "tgt_vocab_file": data_root+"m_common.vocab",
    "source_file": data_root+"wmt14-en-de.src.BPE_common.32K.tok",
    "target_file": data_root+"wmt14-en-de.src.BPE_common.32K.tok",
    "delimiter": " ",
    "shuffle": False,
    "repeat": False,
    "max_length": 256,
  },
}
