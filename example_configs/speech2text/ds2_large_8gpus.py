# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import DeepSpeech2Encoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.data import Speech2TextDataLayer
from open_seq2seq.losses import CTCLoss
from open_seq2seq.optimizers.lr_policies import poly_decay


base_model = Speech2Text

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_gpus": 8,
  "batch_size_per_gpu": 16,

  "num_epochs": 50,

  "save_summaries_steps": 100,
  "print_loss_steps": 10,
  "print_samples_steps": 5000,
  "eval_steps": 5000,
  "save_checkpoint_steps": 1000,
  "logdir": "experiments/librispeech",

  "optimizer": "Momentum",
  "optimizer_params": {
    "momentum": 0.90,
  },
  "lr_policy": poly_decay,
  "lr_policy_params": {
    "learning_rate": 0.001,
    "power": 0.5,
  },
  "larc_params": {
    "larc_eta": 0.001,
  },
  "dtype": tf.float32,
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
    "conv_layers": [
      {
        "kernel_size": [11, 41], "stride": [2, 2],
        "num_channels": 32, "padding": "SAME"
      },
      {
        "kernel_size": [11, 21], "stride": [1, 2],
        "num_channels": 32, "padding": "SAME"
      },
    ],
    "num_rnn_layers": 5,
    "rnn_cell_dim": 800,

    "use_cudnn_rnn": True,
    "rnn_type": "cudnn_gru",
    "rnn_unidirectional": False,

    "row_conv": False,

    "n_hidden": 1600,

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

    "decoder_library_path": "ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
    "lm_path": "language_model/4-gram.binary",
    "trie_path": "language_model/trie.binary",
    "alphabet_config_path": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
  },
  "loss": CTCLoss,
  "loss_params": {},
}

train_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "num_audio_features": 160,
    "input_type": "spectrogram",
    "augmentation": {'time_stretch_ratio': 0.05,
                     'noise_level_min': -90,
                     'noise_level_max': -60},
    "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
    "dataset_files": [
      "data/librispeech/librivox-train-clean-100.csv",
      "data/librispeech/librivox-train-clean-360.csv",
      "data/librispeech/librivox-train-other-500.csv",
    ],
    "max_duration": None,
    "shuffle": True,
  },
}

eval_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "num_audio_features": 160,
    "input_type": "spectrogram",
    "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
    "dataset_files": [
      "data/librispeech/librivox-dev-clean.csv",
    ],
    "shuffle": False,
  },
}
