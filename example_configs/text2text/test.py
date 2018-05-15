from __future__ import absolute_import, division, print_function
from open_seq2seq.models import BasicText2TextWithAttention
from open_seq2seq.encoders import TransformerEncoder
from open_seq2seq.decoders import TransformerDecoder
from open_seq2seq.data.text2text import TransformerDataLayer
from open_seq2seq.losses import PaddedCrossEntropyLossWithSmoothing
from open_seq2seq.optimizers.lr_policies import transformer_policy
import tensorflow as tf

"""
This configuration file describes a tiny variant of Transformer model from
https://arxiv.org/abs/1706.03762 on the toy task of reversing sequences
"""

base_model = BasicText2TextWithAttention
d_model = 512

base_params = {
  "use_horovod": False,
  "num_gpus": 1,
  "batch_size_per_gpu": 2048,
  "max_steps": 5251,
  "save_summaries_steps": 50,
  "print_loss_steps": 50,
  "print_samples_steps": 50,
  "eval_steps": 250,
  "save_checkpoint_steps": 300,
  "logdir": "TEST",

  "optimizer": tf.contrib.opt.LazyAdamOptimizer,
  "optimizer_params": {
    "beta1": 0.9,
    "beta2": 0.997,
    "epsilon": 0.000000001,
  },
  "learning_rate": 2.0,
  "lr_policy": transformer_policy,
  "lr_policy_params": {
    "warmup_steps": 4000,
    "d_model": d_model,
  },
  "dtype": tf.float32,
  "encoder": TransformerEncoder,
  "encoder_params": {
    "encoder_layers": 6,
    "hidden_size": d_model,
    "num_heads": 8,
    "attention_dropout": 0.0,
    "filter_size": 4*d_model,
    "relu_dropout": 0.0,
    "layer_postprocess_dropout": 0.0,
  },

  "decoder": TransformerDecoder,
  "decoder_params": {
    "layer_postprocess_dropout": 0.0,
    "num_hidden_layers": 6,
    "hidden_size": d_model,
    "num_heads": 8,
    "attention_dropout": 0.0,
    "relu_dropout": 0.0,
    "filter_size": 4*d_model,
    "beam_size": 1,
    "alpha": 1.0,
    "extra_decode_length": 50,
  },

  "loss": PaddedCrossEntropyLossWithSmoothing,
  "loss_params": {}
}

train_params = {
  "data_layer": TransformerDataLayer,
  "data_layer_params": {
    'data_dir': "/home/okuchaiev/repos/forks/reference/translation/processed_data/",
    'file_pattern': "*dev*",
    'src_vocab_file': "/home/okuchaiev/repos/forks/reference/translation/processed_data/vocab.ende.32768",
    'batch_size': 2048,
    'max_length': 256,
    'shuffle': True,
    'repeat': 100000,
    'mode': 'train',
    "delimiter": ' ',
  },
}

eval_params = {
  "data_layer": TransformerDataLayer,
  "data_layer_params": {
    'data_dir': "/home/okuchaiev/repos/forks/reference/translation/processed_data/",
    'file_pattern': "*dev*",
    'src_vocab_file': "/home/okuchaiev/repos/forks/reference/translation/processed_data/vocab.ende.32768",
    'batch_size': 2048,
    'max_length': 256,
    'shuffle': False,
    'repeat': 1,
    'mode': 'eval',
    "delimiter": ' ',
  },
}

infer_params = {
  "data_layer": TransformerDataLayer,
  "data_layer_params": {
    'data_dir': "/home/okuchaiev/repos/forks/reference/translation/processed_data/",
    'file_pattern': "*dev*",
    'src_vocab_file': "/home/okuchaiev/repos/forks/reference/translation/processed_data/vocab.ende.32768",
    'batch_size': 2048,
    'max_length': 256,
    'shuffle': False,
    'repeat': 1,
    'mode': 'eval',
    "delimiter": ' ',
  },
}