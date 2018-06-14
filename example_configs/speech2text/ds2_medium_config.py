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
  "num_epochs": 50,

  "num_gpus": 4,
  "batch_size_per_gpu": 32,

  "save_summaries_steps": 100,
  "print_loss_steps": 10,
  "print_samples_steps": 3000,
  "eval_steps": 3000,
  "save_checkpoint_steps": 1000,
  "logdir": "ds2_log_folder",

  "optimizer": "Momentum",
  "optimizer_params": {
    "momentum": 0.90,
  },
  "lr_policy": poly_decay,
  "lr_policy_params": {
    "learning_rate": 0.001,
    "power": 1,
  },
  "dtype": tf.float32,
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 0.0005
  },

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
        "num_channels": 64, "padding": "SAME"
      },
      {
        "kernel_size": [11, 21], "stride": [1, 2],
        "num_channels": 96, "padding": "SAME"
      },
    ],
    "n_hidden": 1024,

    "rnn_cell_dim": 768,
    "rnn_type": "gru",
    "num_rnn_layers": 5,
    "rnn_unidirectional": False,
    "row_conv": False,
    "row_conv_width": 8,
    "use_cudnn_rnn": True,

    "dropout_keep_prob": 0.5,

    "initializer": tf.contrib.layers.xavier_initializer,
    "initializer_params": {
      'uniform': False,
    },
    "activation_fn": lambda x: tf.minimum(tf.nn.relu(x), 20.0),
    "data_format": "channels_first",
  },

  "decoder": FullyConnectedCTCDecoder,
  "decoder_params": {
    "initializer": tf.contrib.layers.xavier_initializer,
    "use_language_model": True,

    # params for decoding the sequence with language model
    "beam_width": 512,
    "lm_weight": 1.0,
    "word_count_weight": 1.5,
    "valid_word_count_weight": 2.5,

    "decoder_library_path": "ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
    "lm_binary_path": "language_model/lm.binary",
    "lm_trie_path": "language_model/trie",
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
    "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
    "dataset_files": [
      "/data/librispeech/librivox-train-clean-100.csv",
      "/data/librispeech/librivox-train-clean-360.csv",
      "/data/librispeech/librivox-train-other-500.csv",
    ],
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
      "/data/librispeech/librivox-dev-clean.csv",
    ],
    "shuffle": False,
  },
}

infer_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "num_audio_features": 160,
    "input_type": "spectrogram",
    "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
    "dataset_files": [
      "/data/librispeech/librivox-test-clean.csv",
    ],
    "shuffle": False,
  },
}