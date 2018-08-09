from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from open_seq2seq.models import Text2Text
from open_seq2seq.data.text2text.text2text import ParallelTextDataLayer
from open_seq2seq.data.text2text.text2text import SpecialTextTokens
from open_seq2seq.data.text2text.tokenizer import EOS_ID

from open_seq2seq.encoders import ConvS2SEncoder, ConvS2SEncoder2
from open_seq2seq.decoders import ConvS2SDecoder, ConvS2SDecoder2

from open_seq2seq.losses import BasicSequenceLoss
from open_seq2seq.optimizers.lr_policies import transformer_policy, poly_decay, fixed_lr
from open_seq2seq.parts.convs2s.utils import gated_linear_units

import math

# REPLACE THIS TO THE PATH WITH YOUR WMT DATA
data_root = "./wmt16_en_dt_old/"

base_model = Text2Text
num_layers = 15
d_model = 512 #512
hidden_before_last = 512 #512 #if shared embedding is used, it should be d_model
max_length = 64
pad_2_eight = True

batch_size = 128
epoch_num = 35
num_gpus = 8

iter_size = 1
dtype = "mixed"  # "mixed" or tf.float32
shuffle_train = True
use_horovod = True

max_steps = int((4500000 / (num_gpus * batch_size * iter_size)) * epoch_num)

conv_act = gated_linear_units #tf.nn.relu tf.nn.tanh gated_linear_units
normalization_type = "weight_norm"  #weight_norm or "batch_norm" or None
#scaling_factor = math.sqrt(0.5) #changed here

base_params = {
  # iter_size can be used just with horovod
  #"iter_size": iter_size,
  "use_horovod": use_horovod,
  "num_gpus": num_gpus,
  "dtype": dtype,

  # set max_step to achieve the given epoch_num, 4.5M is the size of the dataset
  "max_steps": max_steps,
  "batch_size_per_gpu": batch_size,
  "save_summaries_steps": max(1, int(max_steps/1000.0)),
  "print_loss_steps": max(1, int(max_steps/1000.0)),
  "print_samples_steps": None,# max(1, int(max_steps/1000.0)),
  "eval_steps": max(1, int(max_steps/100.0)),
  "save_checkpoint_steps": int((max_steps-1)/5.0),
  "logdir": "WMT16_EN_DT",


  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": transformer_policy,
  "lr_policy_params": {
    "learning_rate": 9,
    "max_lr": 1e-3,
    "warmup_steps": 4000,
    "d_model": d_model,
  },

  # "optimizer": "Momentum",
  # "optimizer_params": {
  #   "momentum": 0.90,
  # },
  #
  # "lr_policy": fixed_lr,
  # "lr_policy": poly_decay,
  # "lr_policy_params": {
  #    "learning_rate": 1.0,
  #    "power": 2.0,
  #    "decay_steps":max_steps,
  # },

  "max_grad_norm": 0.1,

  # "larc_params": {
  #   "larc_eta": 0.00001,
  # },

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm', 'loss_scale'],

  #"regularizer": tf.contrib.layers.l2_regularizer,

  # "regularizer_params": {
  #   "scale": 0.001
  # },

  "loss_scaling": "Backoff",
  # "loss_scaling_params": {
  #    "scale_min": 1.0,
  #    "scale_max": 10., #2.**24., #100., #1024.0, #2.**24,
  #  },

  "encoder": ConvS2SEncoder,
  "encoder_params": {

    "src_emb_size": d_model,
    "pad_embeddings_2_eight": pad_2_eight,
    "att_layer_num": num_layers,

    # original ConvS2S paper
    #"conv_nchannels_kwidth": [(512, 3)]*10 + [(768, 3)]*3 + [(2048, 1)]*2,

    # fairseq config
    "conv_nchannels_kwidth": [(512*2, 3)]*9 + [(1024, 3)]*4 + [(2048, 1)]*2,

    "embedding_dropout_keep_prob": 0.8,
    "hidden_dropout_keep_prob": 0.8,

    "max_input_length": max_length,

    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,

    "conv_activation": conv_act,
    "normalization_type": normalization_type,
    #"scaling_factor": scaling_factor,
  },


  "decoder": ConvS2SDecoder,
  "decoder_params": {

    "shared_embed": True,
    "tgt_emb_size": d_model,
    "pad_embeddings_2_eight": pad_2_eight,
    "out_emb_size": hidden_before_last,
    "pos_embed": False,

    # original ConvS2S paper
    #"conv_nchannels_kwidth": [(512, 3)]*10 + [(768, 3)]*3 + [(2048, 1)]*2,

    # fairseq config
    "conv_nchannels_kwidth": [(512*2, 3)]*9 + [(1024, 3)]*4 + [(2048, 1)]*2,

    "embedding_dropout_keep_prob": 0.8,
    "hidden_dropout_keep_prob": 0.8,
    "out_dropout_keep_prob": 0.8,

    "max_input_length": max_length,
    "extra_decode_length": 56,
    "beam_size": 5,
    "alpha": 0.6,

    "EOS_ID": EOS_ID,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,

    "conv_activation": conv_act,
    "normalization_type": normalization_type,
    #"scaling_factor": scaling_factor,
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
    "pad_vocab_to_eight": pad_2_eight,
    "src_vocab_file": data_root + "vocab.bpe.32000",
    "tgt_vocab_file": data_root + "vocab.bpe.32000",
    "source_file": data_root + "train.tok.clean.bpe.32000.en",
    "target_file": data_root + "train.tok.clean.bpe.32000.de",
    "delimiter": " ",
    "shuffle": shuffle_train,
    "shuffle_buffer_size": 25000,
    "repeat": True,
    "map_parallel_calls": 16,
    "prefetch_buffer_size": 2,
    "max_length": max_length,
  },
}

eval_params = {
  "batch_size_per_gpu": 64,
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "pad_vocab_to_eight": pad_2_eight,
    "src_vocab_file": data_root + "vocab.bpe.32000",
    "tgt_vocab_file": data_root + "vocab.bpe.32000",
    "source_file": data_root + "newstest2014.tok.bpe.32000.en",
    "target_file": data_root + "newstest2014.tok.bpe.32000.de",
    "delimiter": " ",
    "shuffle": False,
    "repeat": True,
    "max_length": max_length,
  },

}

infer_params = {
  "batch_size_per_gpu": 64,
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "pad_vocab_to_eight": pad_2_eight,
    "src_vocab_file": data_root + "vocab.bpe.32000",
    "tgt_vocab_file": data_root + "vocab.bpe.32000",
    "source_file": data_root + "newstest2014.tok.bpe.32000.en",
    # this is intentional to be sure that model is not using target
    "target_file": data_root + "newstest2014.tok.bpe.32000.en",
    "delimiter": " ",
    "shuffle": False,
    "repeat": False,
    "max_length": max_length,
  },
}
