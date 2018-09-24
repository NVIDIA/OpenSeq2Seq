# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import ListenAttendSpellEncoder
from open_seq2seq.decoders import JointCTCAttentionDecoder
from open_seq2seq.data import Speech2TextDataLayer
from open_seq2seq.losses import MultiTaskCTCEntropyLoss
from open_seq2seq.optimizers.lr_policies import poly_decay
from open_seq2seq.decoders import ListenAttendSpellDecoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder

base_model = Speech2Text

base_params = {
    "random_seed": 0,
    "use_horovod": True,
    "num_epochs": 50,

    "num_gpus": 8,
    "batch_size_per_gpu": 64,
    "iter_size": 1,

    "save_summaries_steps": 1100,
    "print_loss_steps": 10,
    "print_samples_steps": 200,
    "eval_steps": 1100,
    "save_checkpoint_steps": 1100,
    "logdir": "jca_log_folder",

    "optimizer": "Adam",
    "optimizer_params": {
    },
    "lr_policy": poly_decay,
    "lr_policy_params": {
        "learning_rate": 1e-3,
        "power": 2.0,
        "min_lr": 1e-5
    },

    "max_grad_norm": 1.0,

    "regularizer": tf.contrib.layers.l2_regularizer,
    "regularizer_params": {
        'scale': 0.0001
    },

    #"dtype": "mixed",
    #"loss_scaling": "Backoff",

    "dtype": tf.float32,

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
                "type": "conv1d", "repeat": 7,
                "kernel_size": [11], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [11], "stride": [2],
                "num_channels": 384, "padding": "SAME",
                "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [11], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 4,
                "kernel_size": [11], "stride": [1],
                "num_channels": 768, "padding": "SAME",
                "dropout_keep_prob": 0.7,
            },
        ],

        "recurrent_layers": [],

        "dropout_keep_prob": 0.8,

        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            'uniform': False,
        },
        "normalization": "batch_norm",
        "activation_fn": lambda x: tf.minimum(tf.nn.relu(x), 20.0),
        "data_format": "channels_last",
    },

    "decoder": JointCTCAttentionDecoder,
    "decoder_params": {

        "attn_decoder": ListenAttendSpellDecoder,
        "attn_decoder_params": {
            "tgt_emb_size": 256,
            "pos_embedding": True,

            "attention_params": {
                "attention_dim": 256,
                "attention_type": "chorowski",
                "use_coverage": True,
                "num_heads": 1,
                "plot_attention": True,

            },

            "rnn_type": "lstm",
            "hidden_dim": 512,
            "num_layers": 1,

            "dropout_keep_prob": 0.8,
        },

        "ctc_decoder": FullyConnectedCTCDecoder,
        "ctc_decoder_params": {
            "initializer": tf.contrib.layers.xavier_initializer,
            "use_language_model": False,
        },

        "beam_search_params": {
            "beam_width": 4,
        },

        "language_model_params": {
            # params for decoding the sequence with language model
            "use_language_model": False,
        },

    },

    "loss": MultiTaskCTCEntropyLoss,
    "loss_params": {

        "seq_loss_params": {
            "offset_target_by_one": False,
            "average_across_timestep": True,
            "do_mask": True
        },

        "ctc_loss_params": {
        },

        "lambda_value": 0.25,
    }
}

train_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            "data/librispeech/librivox-train-clean-100.csv",
            "data/librispeech/librivox-train-clean-360.csv",
            "data/librispeech/librivox-train-other-500.csv",
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
            "data/librispeech/librivox-dev-clean.csv",
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
            "data/librispeech/librivox-test-clean.csv",
        ],
        "shuffle": False,
        "autoregressive": True,
    },
}
