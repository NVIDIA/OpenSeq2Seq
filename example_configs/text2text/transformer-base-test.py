from __future__ import absolute_import, division, print_function
from open_seq2seq.models import Text2Text
from open_seq2seq.encoders import TransformerEncoder
from open_seq2seq.decoders import TransformerDecoder
from open_seq2seq.data.text2text.text2text import TransformerDataLayer
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
d_model = 512
num_layers = 6

data_root = "/tmp/translate_ende/"

base_params = {
  "use_horovod": False,
  "num_gpus": 2,
  "batch_size_per_gpu": 4096,  # this size is in tokens
  "max_steps": 500000,
  "save_summaries_steps": 50,
  "print_loss_steps": 50,
  "print_samples_steps": 50,
  "eval_steps": 4001,
  "save_checkpoint_steps": 1000,
  "logdir": "Transformer-FP32",
  "dtype": tf.float32,
  # "dtype": "mixed",
  # "loss_scaling": "Backoff",
  "optimizer": tf.contrib.opt.LazyAdamOptimizer,
  "optimizer_params": {
    "beta1": 0.9,
    "beta2": 0.997,
    "epsilon": 1e-09,
  },

  "lr_policy": transformer_policy,
  "lr_policy_params": {
    "learning_rate": 2.0,
    "warmup_steps": 16000,
    "d_model": d_model,
  },

  # "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
  #              'variable_norm', 'gradient_norm', 'global_gradient_norm'],

  "encoder": TransformerEncoder,
  "encoder_params": {
    "encoder_layers": num_layers,
    "hidden_size": d_model,
    "num_heads": 8,
    "attention_dropout": 0.1,
    "filter_size": 4 * d_model,
    "relu_dropout": 0.1,
    "layer_postprocess_dropout": 0.1,
    "pad_embeddings_2_eight": True,
  },

  "decoder": TransformerDecoder,
  "decoder_params": {
    "layer_postprocess_dropout": 0.1,
    "num_hidden_layers": num_layers,
    "hidden_size": d_model,
    "num_heads": 8,
    "attention_dropout": 0.1,
    "relu_dropout": 0.1,
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
  "data_layer": TransformerDataLayer,
  "data_layer_params": {
    'data_dir': data_root,
    'file_pattern': "*train*",
    'src_vocab_file': data_root + "vocab.ende.32768",
    'max_length': 256,
    'shuffle': True,
    'repeat': 100000,
    'mode': 'train',
    "delimiter": ' ',
  },
}

eval_params = {
  "batch_size_per_gpu": 256,
  "data_layer": TransformerDataLayer,
  "data_layer_params": {
    'data_dir': data_root,
    'file_pattern': "*dev*",
    'src_vocab_file': data_root + "vocab.ende.32768",
    'max_length': 256,
    'shuffle': False,
    'repeat': 1,
    'mode': 'train',
    "delimiter": ' ',
  },
}

infer_params = {
  "batch_size_per_gpu": 64, # it is now in samples, not tokens
  "data_layer": TransformerDataLayer,
  "data_layer_params": {
    'data_dir': data_root,
    'file_pattern': "*test*",
    'batch_in_tokens': False, # this is necessary to preserve the order
    'src_vocab_file': data_root + "vocab.ende.32768",
    'max_length': 256,
    'shuffle': False,
    'repeat': 1,
    'mode': 'train',
    "delimiter": ' ',
  },
}