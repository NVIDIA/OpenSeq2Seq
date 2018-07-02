from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from open_seq2seq.models import Text2Text
from open_seq2seq.data.text2text.text2text import ParallelTextDataLayer

from open_seq2seq.data.text2text.text2text import SpecialTextTokens
from open_seq2seq.data.text2text.tokenizer import EOS_ID

from open_seq2seq.encoders import ConvS2SEncoder
from open_seq2seq.decoders import ConvS2SDecoder

from open_seq2seq.losses import BasicSequenceLoss, PaddedCrossEntropyLossWithSmoothing

from open_seq2seq.optimizers.lr_policies import transformer_policy

# REPLACE THIS TO THE PATH WITH YOUR WMT DATA
data_root = "./wmt16_en_dt/"

base_model = Text2Text
num_layers = 15
d_model = 768
max_length = 128

batch_size = 64
num_gpus = 4
epoch_num = 30

base_params = {
  "use_horovod": False,
  "num_gpus": num_gpus,
  # set max_step to achieve the given epoch_num, 4.5M is the size of the dataset
  "max_steps": int((4500000 / (num_gpus * batch_size)) * epoch_num),
  "batch_size_per_gpu": batch_size,
  "save_summaries_steps": 50,
  "print_loss_steps": 50,
  "print_samples_steps": 50,
  "eval_steps": 4000,
  "save_checkpoint_steps": 1000,
  "logdir": "RealData-CC",


  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": transformer_policy,
  "lr_policy_params": {
    "learning_rate": 9,
    "max_lr": 1e-3,
    "warmup_steps": 4000,
    "d_model": d_model,
  },

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],


  "max_grad_norm": 0.1,
  "dtype": tf.float32,
  #"dtype": "mixed",
  #"loss_scaling": "Backoff",

  "encoder": ConvS2SEncoder,
  "encoder_params": {
    "encoder_layers": num_layers,

    "src_emb_size": d_model,
    "pad_embeddings_2_eight": False,
    "att_layer_num": num_layers,

    # original paper
    #"conv_nchannels_kwidth": [(512, 3)]*10 + [(768, 3)]*3 + [(2048, 1)]*2,

    # fairseq config
    "conv_nchannels_kwidth": [(512, 3)]*9 + [(1024, 3)]*4 + [(2048, 1)]*2,

    "embedding_dropout_keep_prob": 0.8,
    "hidden_dropout_keep_prob": 0.8,

    "max_input_length": max_length,

    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,
  },


  "decoder": ConvS2SDecoder,
  "decoder_params": {
    "decoder_layers": num_layers,

    "shared_embed": True,
    "tgt_emb_size": d_model,
    "pad_embeddings_2_eight": False,
    "out_emb_size": d_model,

    # original paper
    #"conv_nchannels_kwidth": [(512, 3)]*10 + [(768, 3)]*3 + [(2048, 1)]*2,

    # fairseq config
    "conv_nchannels_kwidth": [(512, 3)]*9 + [(1024, 3)]*4 + [(2048, 1)]*2,

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
    "pad_vocab_to_eight": False,
    "src_vocab_file": data_root + "vocab.bpe.32000",
    "tgt_vocab_file": data_root + "vocab.bpe.32000",
    "source_file": data_root+"train.tok.clean.bpe.32000.en",
    "target_file": data_root+"train.tok.clean.bpe.32000.de",
    "delimiter": " ",
    "shuffle": True,
    "repeat": True,
    "map_parallel_calls": 8,
    "prefetch_buffer_size": 4,
    "max_length": max_length,
  },
}

eval_params = {
  "batch_size_per_gpu": 64,
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "pad_vocab_to_eight": False,
    "src_vocab_file": data_root+"vocab.bpe.32000",
    "tgt_vocab_file": data_root+"vocab.bpe.32000",
    "source_file": data_root+"newstest2014.tok.bpe.32000.en",
    "target_file": data_root+"newstest2014.tok.bpe.32000.de",
    "delimiter": " ",
    "shuffle": False,
    "repeat": True,
    "max_length": 64,
  },

}

infer_params = {
  "batch_size_per_gpu": 1,
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "pad_vocab_to_eight": False,
    "src_vocab_file": data_root+"vocab.bpe.32000",
    "tgt_vocab_file": data_root+"vocab.bpe.32000",
    "source_file": data_root+"newstest2013.tok.bpe.32000.en",
    # this is intentional to be sure that model is not using target
    "target_file": data_root+"newstest2013.tok.bpe.32000.en",
    "delimiter": " ",
    "shuffle": False,
    "repeat": False,
    "max_length": max_length,
  },
}


