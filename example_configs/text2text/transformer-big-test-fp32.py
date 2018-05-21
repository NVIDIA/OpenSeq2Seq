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
This configuration file describes a tiny variant of Transformer model from
https://arxiv.org/abs/1706.03762 on the toy task of reversing sequences
"""

base_model = Text2Text
d_model = 512
num_layers = 6
data_root = "/home/okuchaiev/repos/forks/reference/translation/processed_data/"

base_params = {
  "use_horovod": False,
  "num_gpus": 1,
  "batch_size_per_gpu": 2048,
  "max_steps": 340000,
  "save_summaries_steps": 50,
  "print_loss_steps": 50,
  "print_samples_steps": 50,
  "eval_steps": 4000,
  "save_checkpoint_steps": 300,
  "logdir": "Transformer-big-test-FP32-DL2",
  "dtype": tf.float32,
  "optimizer": tf.contrib.opt.LazyAdamOptimizer,
  "optimizer_params": {
    "beta1": 0.9,
    "beta2": 0.997,
    "epsilon": 1e-09,
  },

  "learning_rate": 2.0,
  "lr_policy": transformer_policy,
  "lr_policy_params": {
    "warmup_steps": 16000,
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
  "batch_size_per_gpu": 512,
  "data_layer": TransformerDataLayer,
  "data_layer_params": {
    'data_dir': data_root,
    'file_pattern': "*dev*",
    'src_vocab_file': data_root + "vocab.ende.32768",
    'max_length': 512,
    'shuffle': False,
    'repeat': 1,
    'mode': 'train',
    "delimiter": ' ',
  },
}
