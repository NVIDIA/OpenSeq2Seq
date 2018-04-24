from __future__ import absolute_import, division, print_function
from open_seq2seq.models import BasicText2TextWithAttention
from open_seq2seq.encoders import TransformerEncoder
from open_seq2seq.decoders import TransformerDecoder
from open_seq2seq.data.text2text import ParallelTextDataLayer
from open_seq2seq.losses import BasicSequenceLoss
from open_seq2seq.data.text2text import SpecialTextTokens
from open_seq2seq.optimizers.lr_policies import transformer_policy
import tensorflow as tf

data_root = "[REPLACE THIS TO THE PATH WITH YOUR WMT DATA]"

# This model is work in progress
base_model = BasicText2TextWithAttention

base_params = {
  "use_horovod": False,
  "num_gpus": 4,
  "max_steps": 150000,
  "batch_size_per_gpu": 256,
  "save_summaries_steps": 50,
  "print_loss_steps": 50,
  "print_samples_steps": 50,
  "eval_steps": 1000,
  "save_checkpoint_steps": 2001,
  "logdir": "TransformerSmall-En-De",

  "optimizer": "Adam",
  "optimizer_params": {
    "beta1": 0.9,
    "beta2": 0.997,
    "epsilon": 0.000000001,
  },
  "learning_rate": 1.0,
  "lr_policy": transformer_policy,
  "lr_policy_params": {
    "warmup_steps": 8000,
    "d_model": 256,
  },
  "dtype": tf.float32,

  "encoder": TransformerEncoder,
  "encoder_params": {
    "initializer": tf.glorot_uniform_initializer,
    "d_model": 256,
    "ffn_inner_dim": 1024,
    "encoder_layers": 2,
    "attention_heads": 4,
    "encoder_drop_prob": 0.0,
  },

  "decoder": TransformerDecoder,
  "decoder_params": {
    "initializer": tf.glorot_uniform_initializer,
    "use_encoder_emb": True,
    "tie_emb_and_proj": True,
    "d_model": 256,
    "ffn_inner_dim": 1024,
    "decoder_layers": 2,
    "attention_heads": 4,
    "decoder_drop_prob": 0.0,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,
  },

  #"loss": CrossEntropyWithSmoothing,
  "loss": BasicSequenceLoss,
  "loss_params": {
    "offset_target_by_one": True,
    "do_mask": True,
    "average_across_timestep": True,
    #"label_smoothing": 0.1,
  }
}

train_params = {
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "src_vocab_file": data_root+"vocab.bpe.32000",
    "tgt_vocab_file": data_root+"vocab.bpe.32000",
    "source_file": data_root+"train.tok.clean.bpe.32000.en",
    "target_file": data_root+"train.tok.clean.bpe.32000.de",
    "delimiter": " ",
    "shuffle": True,
    "repeat": True,
    "map_parallel_calls": 16,
    "prefetch_buffer_size": 8,
    "max_length": 56,
    "pad_lengths_to_eight": True,
  },
}

eval_params = {
  "batch_size_per_gpu": 16,
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "src_vocab_file": data_root+"vocab.bpe.32000",
    "tgt_vocab_file": data_root+"vocab.bpe.32000",
    "source_file": data_root+"newstest2013.tok.bpe.32000.en",
    "target_file": data_root+"newstest2013.tok.bpe.32000.de",
    "delimiter": " ",
    "shuffle": False,
    "repeat": True,
    "map_parallel_calls": 16,
    "prefetch_buffer_size": 16,
    "max_length": 32,
    "pad_lengths_to_eight": True,
  },
}

infer_params = {
  "batch_size_per_gpu": 1,
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "src_vocab_file": data_root+"vocab.bpe.32000",
    "tgt_vocab_file": data_root+"vocab.bpe.32000",
    "source_file": data_root+"newstest2014.tok.bpe.32000.en",
    # this is intentional
    "target_file": data_root+"newstest2014.tok.bpe.32000.en",
    "delimiter": " ",
    "shuffle": False,
    "repeat": False,
    "pad_lengths_to_eight": True,
    "max_length": 256,
  },
}
