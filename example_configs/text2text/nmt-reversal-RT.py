from __future__ import absolute_import, division, print_function
from open_seq2seq.models import BasicText2TextWithAttention
from open_seq2seq.encoders import BidirectionalRNNEncoderWithEmbedding
from open_seq2seq.decoders import TransformerDecoder
from open_seq2seq.data.text2text import ParallelTextDataLayer
from open_seq2seq.losses import CrossEntropyWithSmoothing
from open_seq2seq.data.text2text import SpecialTextTokens
from open_seq2seq.optimizers.lr_policies import transformer_policy
import tensorflow as tf

"""
This configuration file describes a model which uses RNN-based encoder
and Transformer-based decoder on the toy task of reversing sequences
"""

base_model = BasicText2TextWithAttention

base_params = {
  "use_horovod": False,
  "num_gpus": 1,
  "batch_size_per_gpu": 64,
  "max_steps": 1100,
  "save_summaries_steps": 10,
  "print_loss_steps": 10,
  "print_samples_steps": 20,
  "eval_steps": 50,
  "save_checkpoint_steps": 300,

  "logdir": "ReversalTask-RNN-Transformer",

  "optimizer": "Adam",
  "optimizer_params": {
    "beta1": 0.9,
    "beta2": 0.98,
    "epsilon": 0.000000001,
  },
  "learning_rate": 1.0,
  "lr_policy": transformer_policy,
  "lr_policy_params": {
    "warmup_steps": 600,
    "d_model": 64,
  },
  "dtype": tf.float32,

  "encoder": BidirectionalRNNEncoderWithEmbedding,
  "encoder_params": {
    "encoder_cell_type": "lstm",
    "encoder_cell_units": 64,
    "encoder_layers": 1,
    "encoder_dp_input_keep_prob": 0.8,
    "encoder_dp_output_keep_prob": 1.0,
    "encoder_use_skip_connections": False,
    "src_emb_size": 64,
  },

  "decoder": TransformerDecoder,
  "decoder_params": {
    "initializer": tf.glorot_uniform_initializer,
    "use_encoder_emb": False,
    "tie_emb_and_proj": True,
    "d_model": 64,
    "ffn_inner_dim": 128,
    "decoder_layers": 1,
    "attention_heads": 8,
    "decoder_drop_prob": 0.8,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,
  },

  "loss": CrossEntropyWithSmoothing,
  "loss_params": {
    "offset_target_by_one": True,
    "do_mask": True,
    "label_smoothing": 0.01,
  }
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
  },
}