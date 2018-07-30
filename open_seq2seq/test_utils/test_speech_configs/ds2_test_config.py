# pylint: skip-file
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import DeepSpeech2Encoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.data import Speech2TextDataLayer
from open_seq2seq.losses import CTCLoss
from open_seq2seq.optimizers.lr_policies import poly_decay


base_model = Speech2Text

base_params = {
    "use_horovod": False,
    "num_epochs": 150,

    "num_gpus": 1,
    "batch_size_per_gpu": 10,
    "save_summaries_steps": 10,
    "print_loss_steps": 10,
    "print_samples_steps": 20,
    "eval_steps": 50,
    "save_checkpoint_steps": 50,
    "logdir": "tmp_log_folder",

    "optimizer": "Momentum",
    "optimizer_params": {
        "momentum": 0.90,
    },
    "lr_policy": poly_decay,
    "lr_policy_params": {
        "learning_rate": 0.01,
        "power": 2,
    },
    "larc_params": {
        "larc_eta": 0.001,
    },
    "dtype": tf.float32,
    "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                  'variable_norm', 'gradient_norm', 'global_gradient_norm'],

    "encoder": DeepSpeech2Encoder,
    "encoder_params": {
        "conv_layers": [
            {
                "kernel_size": [5, 11], "stride": [2, 2],
                "num_channels": 32, "padding": "SAME"
            },
            {
                "kernel_size": [5, 11], "stride": [1, 2],
                "num_channels": 64, "padding": "SAME"
            },
        ],
        "n_hidden": 128,

        "rnn_cell_dim": 128,
        "rnn_type": "gru",
        "num_rnn_layers": 1,
        "rnn_unidirectional": False,
        "row_conv": True,
        "row_conv_width": 8,
        "use_cudnn_rnn": True,

        "dropout_keep_prob": 0.9,

        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            'uniform': False,
        },
        "activation_fn": lambda x: tf.minimum(tf.nn.relu(x), 20.0),
        "data_format": "channels_first",
        "bn_momentum": 0.001,
    },

    "decoder": FullyConnectedCTCDecoder,
    "decoder_params": {
        "initializer": tf.contrib.layers.xavier_initializer,
        "use_language_model": False,
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
            "open_seq2seq/test_utils/toy_speech_data/toy_data.csv",
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
            "open_seq2seq/test_utils/toy_speech_data/toy_data.csv",
        ],
        "shuffle": False,
    },
}
