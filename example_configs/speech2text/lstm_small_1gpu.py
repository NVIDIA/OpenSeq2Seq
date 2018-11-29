# pylint: skip-file
import os
import tensorflow as tf

from open_seq2seq.data import Speech2TextDataLayer
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.encoders import DeepSpeech2Encoder
from open_seq2seq.losses import CTCLoss
from open_seq2seq.models import Speech2Text
from open_seq2seq.optimizers.lr_policies import exp_decay


base_model = Speech2Text
dataset_location = os.path.expanduser("~/datasets/speech/librispeech/")

### INPUT FEATURES CONFIG ####
# input_type = "spectrogram"
# num_audio_features = 96

input_type = "mfcc"
num_audio_features = 13  # primary MFCC coefficients


### PREPROCESSING CACHING CONFIG ###
train_cache_features = False
eval_cache_features = True
cache_format = 'hdf5'
cache_regenerate = False

### RNN CONFIG ####
num_rnn_layers = 2
rnn_cell_dim = 512
rnn_type = "cudnn_lstm"
rnn_unidirectional = True


base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_epochs": 50,

  "num_gpus": 1,
  "batch_size_per_gpu": 64,

  "save_summaries_steps": 100,
  "print_loss_steps": 50,
  "print_samples_steps": 250,
  "eval_steps": 250,
  "save_checkpoint_steps": 250,
  "logdir": "logs/librispeech-" +
            rnn_type + str(num_rnn_layers) + "x" + str(rnn_cell_dim) + "-" +
            input_type + str(num_audio_features),

  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": exp_decay,
  "lr_policy_params": {
    "learning_rate": 0.001,
    "begin_decay_at": 0,
    "decay_steps": 500,
    "decay_rate": 0.9,
    "use_staircase_decay": True,
    "min_lr": 1e-8,
  },
  # "dtype": tf.float32,
  "dtype": "mixed",
  "max_grad_norm": 0.25,
  "loss_scaling": "Backoff",

  # weight decay
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 0.0005
  },
  "initializer": tf.contrib.layers.xavier_initializer,

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],

  "encoder": DeepSpeech2Encoder,
  "encoder_params": {

    # CONV layers
    "conv_layers": [  # no CONV layers needed? with MFCC input
    ],

    # RNN layers
    "num_rnn_layers": num_rnn_layers,
    "rnn_cell_dim": rnn_cell_dim,

    "use_cudnn_rnn": True if 'cudnn' in rnn_type else False,
    "rnn_type": rnn_type,
    "rnn_unidirectional": rnn_unidirectional,

    "row_conv": False,

    # FC layers
    "n_hidden": 512,

    "dropout_keep_prob": 0.5,
    "activation_fn": tf.nn.relu,
    "data_format": "channels_first",
  },

  "decoder": FullyConnectedCTCDecoder,
  "decoder_params": {
    "use_language_model": False,
    # params for decoding the sequence with language model
    "beam_width": 512,
    "alpha": 2.0,
    "beta": 1.0,

    "decoder_library_path":
      "ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
    "lm_path": "language_model/4-gram.binary",
    "trie_path": "language_model/trie.binary",
    "alphabet_config_path":
      "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
  },
  "loss": CTCLoss,
  "loss_params": {},
}

train_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "cache_features": train_cache_features,
    "cache_format": cache_format,
    "cache_regenerate": cache_regenerate,
    "num_audio_features": num_audio_features,
    "input_type": input_type,

    "augmentation": {
      'time_stretch_ratio': 0.05,
      'noise_level_min': -90,
      'noise_level_max': -60
    },
    "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
    "dataset_files": [
      os.path.join(dataset_location, "librivox-train-clean-100.csv"),
      os.path.join(dataset_location, "librivox-train-clean-360.csv"),
    ],
    "max_duration": 16.7,
    "shuffle": True,
  },
}

eval_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "cache_features": eval_cache_features,
    "cache_format": cache_format,
    "cache_regenerate": cache_regenerate,
    "num_audio_features": num_audio_features,
    "input_type": input_type,

    "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
    "dataset_files": [
      os.path.join(dataset_location, "librivox-dev-clean.csv"),
    ],
    "shuffle": False,
  },
}
