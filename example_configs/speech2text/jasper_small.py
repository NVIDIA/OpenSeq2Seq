# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import TDNNEncoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.data.speech2text.speech2text import Speech2TextDataLayer
from open_seq2seq.losses import CTCLoss
from open_seq2seq.optimizers.lr_policies import poly_decay
from open_seq2seq.optimizers.novograd import NovoGrad

residual_dense = False # Enable or disable Dense Residual
layer_per_block = 5

data_dir = "/home/lab/"

base_model = Speech2Text
base_params = {
    "random_seed": 0,
    "use_horovod": False,
    "max_steps": 100,
#    "num_epochs": 1,

    "num_gpus": 1,
    "batch_size_per_gpu": 2,
    "iter_size": 1,

    "save_summaries_steps": 100,
    "print_loss_steps": 1,
    "print_samples_steps": 2200,
    "eval_steps": 2200,
    "save_checkpoint_steps": 1100,
    "logdir": "logs/jasper_small2",
    "num_checkpoints": 1,

    "optimizer": NovoGrad,
    "optimizer_params": {
        "beta1": 0.95,
        "beta2": 0.98,
        "epsilon": 1e-08,
        "weight_decay": 0.001,
        "grad_averaging": False,
    },
    "lr_policy": poly_decay,
    "lr_policy_params": {
        "learning_rate": 0.001,
        "min_lr": 1e-5,
        "power": 2.0,
    },

    "dtype": tf.float32,

    "summaries": ['learning_rate', 'variables', 'gradients', 
          'variable_norm', 'gradient_norm', 'global_gradient_norm'],

    "encoder": TDNNEncoder,
    "encoder_params": {
        "convnet_layers": [
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [11], "stride": [2],
                "num_channels": 32, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 1.0, # 0.8,
            },
            {
                "type": "conv1d", "repeat": layer_per_block,
                "kernel_size": [13], "stride": [1],
                "num_channels": 64, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 1.0, # 0.8,
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": layer_per_block,
                "kernel_size": [17], "stride": [1],
                "num_channels": 96, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 1.0, # 0.8,
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": layer_per_block,
                "kernel_size": [21], "stride": [1],
                "num_channels": 128, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 1.0, # 0.7,
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": layer_per_block,
                "kernel_size": [25], "stride": [1],
                "num_channels": 160, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 1.0, # 0.7,
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [29], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dilation":[2], "dropout_keep_prob": 1.0, # 0.6,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [1], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 1.0, # 0.6,
            }
        ],

        "dropout_keep_prob": 1.0, # 0.7,

        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            'uniform': False,
        },
        "normalization": "batch_norm",
        "activation_fn": tf.nn.relu,
        "data_format": "channels_last",
        "use_conv_mask": True,
    },

    "decoder": FullyConnectedCTCDecoder,
    "decoder_params": {
        "initializer": tf.contrib.layers.xavier_initializer,
        "use_language_model": False,

        # params for decoding the sequence with language model
        # "beam_width": 2048,
        # "alpha": 2.0,
        # "beta": 1.5,

        # "decoder_library_path": "ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
        # "lm_path": "language_model/4-gram.binary",
        # "trie_path": "language_model/trie.binary",
        # "alphabet_config_path": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",

        "infer_logits_to_pickle": False,
    },
    "loss": CTCLoss,
    "loss_params": {},

    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "norm_per_feature": True,
        "window": "hanning",
        "precompute_mel_basis": True,
        "sample_freq": 16000,
        "pad_to": 16,
        "backend": "librosa"
    },
}

train_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "dataset_files": [
          data_dir + "data/librispeech/librivox-dev-clean.csv"
#            data_dir + "data/librispeech/librivox-train-clean-100.csv" #,
#            "data/librispeech/librivox-train-clean-360.csv",
#            "data/librispeech/librivox-train-other-500.csv"
        ],
        "max_duration": 4.0,
        "shuffle": False,
    },
}

eval_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "dataset_files": [
            data_dir + "data/librispeech/librivox-dev-clean.csv",
        ],
        "shuffle": False,
    },
}

infer_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "dataset_files": [
            data_dir + "data/librispeech/librivox-test-clean.csv",
        ],
        "shuffle": False,
    },
}
