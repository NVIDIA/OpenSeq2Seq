from __future__ import absolute_import, division, print_function
import tensorflow as tf
from open_seq2seq.models import BasicText2TextWithAttention
from open_seq2seq.encoders import BidirectionalRNNEncoderWithEmbedding
from open_seq2seq.decoders import RNNDecoderWithAttention, \
  BeamSearchRNNDecoderWithAttention
from open_seq2seq.data.text2text.text2text import TransformerDataLayer
from open_seq2seq.losses import BasicSequenceLoss
from open_seq2seq.data.text2text.text2text import SpecialTextTokens

"""
This configuration file describes classic RNN-based encoder-decoder model
with attention on the toy task of reversing sequences
"""

base_model = BasicText2TextWithAttention

base_params = {
  "use_horovod": False,
  # set this to number of available GPUs
  "num_gpus": 1,
  "batch_size_per_gpu": 64,
  "max_steps": 800,
  "save_summaries_steps": 10,
  "print_loss_steps": 10,
  "print_samples_steps": 20,
  "eval_steps": 50,
  "save_checkpoint_steps": 300,
  "logdir": "TEST",

  "optimizer": "Adam",
  "optimizer_params": {"epsilon": 1e-4},
  "learning_rate": 0.001,
  "max_grad_norm": 3.0,
  "dtype": tf.float32,

  "encoder": BidirectionalRNNEncoderWithEmbedding,
  "encoder_params": {
    "encoder_cell_type": "lstm",
    "encoder_cell_units": 128,
    "encoder_layers": 1,
    "encoder_dp_input_keep_prob": 0.8,
    "encoder_dp_output_keep_prob": 1.0,
    "encoder_use_skip_connections": False,
    "src_emb_size": 128,
  },

  "decoder": RNNDecoderWithAttention,
  "decoder_params": {
    "decoder_cell_type": "lstm",
    "decoder_cell_units": 128,
    "decoder_layers": 1,
    "decoder_dp_input_keep_prob": 0.8,
    "decoder_dp_output_keep_prob": 1.0,
    "decoder_use_skip_connections": False,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "tgt_emb_size": 128,
    "attention_type": "luong",
    "luong_scale": False,
    "attention_layer_size": 128,
  },

  "loss": BasicSequenceLoss,
  "loss_params": {
    "offset_target_by_one": True,
    "average_across_timestep": False,
    "do_mask": True
  }
}

train_params = {
  "data_layer": TransformerDataLayer,
  "data_layer_params": {
    'data_dir': "/home/okuchaiev/repos/forks/reference/translation/processed_data/",
    'file_pattern': "*dev*",
    'src_vocab_file': "/home/okuchaiev/repos/forks/reference/translation/processed_data/vocab.ende.32768",
    'batch_size': 512,
    'max_length': 256,
    'shuffle': True,
    'repeat': 10,
  },
}

eval_params = {
  "data_layer": TransformerDataLayer,
  "data_layer_params": {
    'data_dir': "/home/okuchaiev/repos/forks/reference/translation/processed_data/",
    'file_pattern': "*dev*",
    'src_vocab_file': "/home/okuchaiev/repos/forks/reference/translation/processed_data/vocab.ende.32768",
    'batch_size': 512,
    'max_length': 256,
    'shuffle': True,
    'repeat': 1,
  },
}
