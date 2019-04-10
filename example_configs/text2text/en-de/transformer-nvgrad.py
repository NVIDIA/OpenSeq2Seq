# pylint: skip-file
from __future__ import absolute_import, division, print_function
from open_seq2seq.models import Text2Text
from open_seq2seq.encoders import TransformerEncoder
from open_seq2seq.decoders import TransformerDecoder
from open_seq2seq.data.text2text.text2text import ParallelTextDataLayer
from open_seq2seq.losses import PaddedCrossEntropyLossWithSmoothing
from open_seq2seq.data.text2text.text2text import SpecialTextTokens
from open_seq2seq.data.text2text.tokenizer import EOS_ID
from open_seq2seq.optimizers.lr_policies import transformer_policy, poly_decay
from open_seq2seq.optimizers.novograd  import NovoGrad
import tensorflow as tf


"""
This configuration file describes a variant of Transformer model from
https://arxiv.org/abs/1706.03762
"""

base_model = Text2Text
d_model = 1024
num_layers = 6

norm_params= {
  "type":  "layernorm_L2",
  "momentum":0.95,
  "epsilon": 0.00001,
}

attention_dropout = 0.1
dropout = 0.3

# REPLACE THIS TO THE PATH WITH YOUR WMT DATA
#data_root = "[REPLACE THIS TO THE PATH WITH YOUR WMT DATA]"
data_root = "/data/wmt16-ende-sp/"

base_params = {
  "use_horovod": True,
  "num_gpus": 1, #8, # when using Horovod we set number of workers with params to mpirun
  "batch_size_per_gpu": 128,  # this size is in sentence pairs, reduce it if you get OOM
  "max_steps":  600000,
  "save_summaries_steps": 100,
  "print_loss_steps": 100,
  "print_samples_steps": 10000,
  "eval_steps": 10000,
  "save_checkpoint_steps": 99999,
  "logdir": "tr-nvgrad2_0.90.99-b128-lr0.1-fp16",
  # "dtype": tf.float32, # to enable mixed precision, comment this line and uncomment two below lines
  "dtype": "mixed",
  "loss_scaling": "Backoff",

  "optimizer": NovoGrad,
  "optimizer_params": {
    "beta1": 0.95,
    "beta2": 0.99,
    "epsilon":  1e-08,
    "weight_decay": 0.00001,
    "grad_averaging": False,
  },
  "lr_policy": poly_decay,
  "lr_policy_params": {
    "learning_rate": 0.04,
    "power": 2,
  },

  "encoder": TransformerEncoder,
  "encoder_params": {
    "encoder_layers": num_layers,
    "hidden_size": d_model,
    "num_heads": 16,
    "filter_size": 4 * d_model,
    "attention_dropout": attention_dropout,  # 0.1,
    "relu_dropout": dropout,                 # 0.3,
    "layer_postprocess_dropout": dropout,    # 0.3,
    "pad_embeddings_2_eight": True,
    "remove_padding": True,
    "norm_params": norm_params,
  },

  "decoder": TransformerDecoder,
  "decoder_params": {
    "num_hidden_layers": num_layers,
    "hidden_size": d_model,
    "num_heads": 16,
    "filter_size": 4 * d_model,
    "attention_dropout": attention_dropout,  # 0.1,
    "relu_dropout": dropout,                 # 0.3,
    "layer_postprocess_dropout": dropout,    # 0.3,
    "beam_size": 4,
    "alpha": 0.6,
    "extra_decode_length": 50,
    "EOS_ID": EOS_ID,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,
    "norm_params": norm_params,
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
    "repeat": True,
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
