from __future__ import absolute_import, division, print_function
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import DeepSpeech2Encoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.data import Speech2TextTFDataLayer
from open_seq2seq.losses import CTCLoss
from open_seq2seq.optimizers.lr_policies import exp_decay


base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_epochs": 100,

  "num_gpus": 4,
  "batch_size_per_gpu": 32,

  "summary_frequency": 100,
  "print_loss_frequency": 10,
  "print_samples_frequency": 5000,
  "eval_frequency": 5000,
  "checkpoint_frequency": 1000,
  "logdir": "experiments/librispeech",

  "base_model": Speech2Text,
  "model_params": {
    "optimizer": "Adam",
    "optimizer_params": {},
    "learning_rate": 0.0001,
    "lr_policy": exp_decay,
    "lr_policy_params": {
      "begin_decay_at": 0,
      "decay_steps": 5000,
      "decay_rate": 0.9,
      "use_staircase_decay": True,
      "min_lr": 0.0,
    },
    "dtype": tf.float32,
    # weight decay
    "regularizer": tf.contrib.layers.l2_regularizer,
    "regularizer_params": {
      'scale': 0.0005
    },
    "initializer": tf.contrib.layers.xavier_initializer,

    "summaries": ['learning_rate', 'variables', 'gradients',
                  'variable_norm', 'gradient_norm', 'global_gradient_norm']
  },

  "encoder": DeepSpeech2Encoder,
  "encoder_params": {
    "conv_layers": [
      {
        "kernel_size": [11, 41], "stride": [2, 2],
        "num_channels": 32, "padding": "SAME"
      },
      {
        "kernel_size": [11, 21], "stride": [1, 2],
        "num_channels": 64, "padding": "SAME"
      },
      {
        "kernel_size": [11, 21], "stride": [1, 2],
        "num_channels": 96, "padding": "SAME"
      },

    ],
    "num_rnn_layers": 2,
    "rnn_cell_dim": 1024,

    "use_cudnn_rnn": True,
    "rnn_type": "cudnn_gru",
    "rnn_unidirectional": True,

    "row_conv": True,
    "row_conv_width": 8,

    "n_hidden": 2048,

    "dropout_keep_prob": 0.5,
    "activation_fn": tf.nn.relu,
    "data_format": "channels_first",
  },

  "decoder": FullyConnectedCTCDecoder,
  "decoder_params": {
    "use_language_model": True,

    # params for decoding the sequence with language model
    "beam_width": 512,
    "lm_weight": 2.0,
    "word_count_weight": 1.0,
    "valid_word_count_weight": 2.5,

    "decoder_library_path": "ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
    "lm_binary_path": "language_model/lm.binary",
    "lm_trie_path": "language_model/trie",
    "alphabet_config_path": "open_seq2seq/test_utils/toy_speech_data/alphabet.txt",
  },
  "loss": CTCLoss,
  "loss_params": {},
}

train_params = {
  "data_layer": Speech2TextTFDataLayer,
  "data_layer_params": {
    "num_audio_features": 160,
    "input_type": "spectrogram",
    "augmentation": {'time_stretch_ratio': 0.05,
                     'noise_level_min': -90,
                     'noise_level_max': -60},
    "alphabet_config_path": "open_seq2seq/test_utils/toy_speech_data/alphabet.txt",
    "dataset_path": [
      "data/librispeech/librivox-train-clean-100.csv",
      "data/librispeech/librivox-train-clean-360.csv",
      "data/librispeech/librivox-train-other-500.csv"
    ],
    "shuffle": True,
  },
}

eval_params = {
  "data_layer": Speech2TextTFDataLayer,
  "data_layer_params": {
    "num_audio_features": 160,
    "input_type": "spectrogram",
    "alphabet_config_path": "open_seq2seq/test_utils/toy_speech_data/alphabet.txt",
    "dataset_path": [
      "data/librispeech/librivox-dev-clean.csv"
    ],
    "shuffle": False,
  },
}
