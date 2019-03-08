# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import TDNNEncoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.data.speech2text.speech2text import Speech2TextDataLayer
from open_seq2seq.losses import CTCLoss
from open_seq2seq.optimizers.lr_policies import poly_decay

### If training with synthetic data, don't forget to add your synthetic csv
### to dataset files

base_model = Speech2Text

base_params = {
    "use_horovod": False,
    "num_epochs": 500,

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

    "encoder": TDNNEncoder,
    "encoder_params": {
        "convnet_layers": [
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [7], "stride": [1],
                "num_channels": 128, "padding": "SAME",
                "dilation":[1]
            },
            {
                "type": "conv1d", "repeat": 2,
                "kernel_size": [7], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1],
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 2,
                "kernel_size": [1], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1],
                "residual": True
            },
        ],

        "dropout_keep_prob": 0.9,

        "drop_block_prob": 0.2,
        "drop_block_index": -1,

        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            'uniform': False,
        },
        "normalization": "batch_norm",
        "activation_fn": lambda x: tf.minimum(tf.nn.relu(x), 20.0),
        "data_format": "channels_last",
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
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            "/data/librispeech/librivox-train-clean-100.csv",
            "/data/librispeech/librivox-train-clean-360.csv",
            "/data/librispeech/librivox-train-other-500.csv",
            # Add synthetic csv here
        ],
        "syn_enable": False, # Change to True if using synthetic data
        "syn_subdirs": [], # Add subdirs of synthetic data
        "max_duration": 16.7,
        "shuffle": True,
    },
}

train_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 40,
        "input_type": "logfbank",
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
        "num_audio_features": 40,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            "open_seq2seq/test_utils/toy_speech_data/toy_data.csv",
        ],
        "shuffle": False,
    },
}
