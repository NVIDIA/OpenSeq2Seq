# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import ListenAttendSpellEncoder
from open_seq2seq.decoders import ListenAttendSpellDecoder
from open_seq2seq.data import Speech2TextDataLayer
from open_seq2seq.losses import BasicSequenceLoss
from open_seq2seq.optimizers.lr_policies import poly_decay


base_model = Speech2Text

base_params = {
    "random_seed": 0,
    "use_horovod": False,
    "num_epochs": 50,

    "num_gpus": 8,
    "batch_size_per_gpu": 64,
    "iter_size": 1,

    "save_summaries_steps": 100,
    "print_loss_steps": 10,
    "print_samples_steps": 100,
    "eval_steps": 200,
    "save_checkpoint_steps": 1100,
    "logdir": "las_log_folder",

    "optimizer": "Adam",
    "optimizer_params": {
    },
    "lr_policy": poly_decay,
    "lr_policy_params": {
        "learning_rate": 1e-3,
        "power": 2.0,
        "min_lr": 1e-5
    },

    "larc_params": {
        "larc_eta": 0.001,
    },

    "regularizer": tf.contrib.layers.l2_regularizer,
    "regularizer_params": {
        'scale': 0.0001
    },

    "dtype": "mixed",
    "loss_scaling": "Backoff",

    "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                  'variable_norm', 'gradient_norm', 'global_gradient_norm'],

    "encoder": ListenAttendSpellEncoder,
    "encoder_params": {

        "convnet_layers": [
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [11], "stride": [2],
                "num_channels": 256, "padding": "SAME",
                "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [11], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [13], "stride": [2],
                "num_channels": 384, "padding": "SAME",
                "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [13], "stride": [1],
                "num_channels": 384, "padding": "SAME",
                "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [17], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [17], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [21], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dropout_keep_prob": 0.7,
            },
            {
                "type": "conv1d", "repeat": 2,
                "kernel_size": [25], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dropout_keep_prob": 0.7,
            },
        ],

        "recurrent_layers": [],
        "dropout_keep_prob": 0.8,

        "residual_connections": False,

        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            'uniform': False,
        },
        "normalization": "batch_norm",
        "activation_fn": lambda x: tf.minimum(tf.nn.relu(x), 20.0),
        "data_format": "channels_last",
    },

    "decoder": ListenAttendSpellDecoder,
    "decoder_params": {
        "tgt_emb_size": 256,

        "pos_embedding": False,

        "attention_params": {
            "attention_dim": 256,
            "attention_type": "bahadanuwithlocation",
            "use_coverage": False,
        },
        
        "rnn_type": "lstm",
        "hidden_dim": 512,
        "num_layers": 1,

        "dropout_keep_prob": 1.0,
    },

    "loss": BasicSequenceLoss,
    "loss_params": {
        "offset_target_by_one": False,
        "average_across_timestep": True,
        "do_mask": True
    }
}

train_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            "../librispeech/librivox-train-clean-100.csv",
            "../librispeech/librivox-train-clean-360.csv",
            "../librispeech/librivox-train-other-500.csv",
        ],
        "max_duration": 16.7,
        "shuffle": True,
        "autoregressive": True,
    },
}

eval_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            "/data/librispeech/librivox-dev-clean.csv",
        ],
        "shuffle": False,
        "autoregressive": True,
    },
}

infer_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            "/data/librispeech/librivox-test-clean.csv",
        ],
        "shuffle": False,
        "autoregressive": True,
    },
}
