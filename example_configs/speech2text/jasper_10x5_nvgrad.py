# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import TDNNEncoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.data.speech2text.speech2text import Speech2TextDataLayer
from open_seq2seq.losses import CTCLoss
from open_seq2seq.optimizers.lr_policies import poly_decay
from open_seq2seq.optimizers.novograd  import NovoGrad

### If training with synthetic data, don't forget to add your synthetic csv
### to dataset files

# REPLACE THIS TO THE PATH TO LIBRISPEECH
#data_root = "[REPLACE THIS TO THE PATH TO LIBRISPEECH]"
# data_root = "/raid/speech/librispeech/"
# data_root = "/raid/speech/librispeech/"
data_root ="/data/librispeech/"

base_model = Speech2Text

base_params = {
    "random_seed": 0,
    "use_horovod": True,
    "num_epochs": 400,
#     "max_steps": 10000, #000,
    "num_gpus": 1,
    "batch_size_per_gpu": 32,
    "iter_size": 1,

    "save_summaries_steps": 100,
    "print_loss_steps": 100,
    "print_samples_steps": 20000,
    "eval_steps": 10000,
    "save_checkpoint_steps": 10000,
    "num_checkpoints": 1,
    "logdir": "jasper_nvgd_lr0.02p2_wd0.001_fp16",

    "optimizer": NovoGrad,
    "optimizer_params": {
        "beta1": 0.95,
        "beta2": 0.98,
        "epsilon": 1e-08,
        "weight_decay": 0.001,
        "grad_averaging": False,
    },
    "lr_policy": poly_decay,  # fixed_lr,
    "lr_policy_params": {
        "learning_rate": 0.02,  #
        "power": 2.0,
        # "warmup_steps": 200,
    },

    # "optimizer": "Momentum",
    # "optimizer_params": {
    #     "momentum": 0.90,
    # },
    # "lr_policy": poly_decay,
    # "lr_policy_params": {
    #     "learning_rate": 0.01,
    #     "min_lr": 1e-5,
    #     "power": 2.0,
    # },
    # "larc_params": {
    #     "larc_eta": 0.001,
    # },

    # "regularizer": tf.contrib.layers.l2_regularizer,
    # "regularizer_params": {
    #     'scale': 0.001
    # },

    "dtype": "mixed",
    "loss_scaling": "Backoff",

    "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                  'variable_norm', 'gradient_norm', 'global_gradient_norm'],

    "encoder": TDNNEncoder,
    "encoder_params": {
        "convnet_layers": [
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [11], "stride": [2],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [11], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [11], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [13], "stride": [1],
                "num_channels": 384, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [13], "stride": [1],
                "num_channels": 384, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [17], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [17], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [21], "stride": [1],
                "num_channels": 640, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [21], "stride": [1],
                "num_channels": 640, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [25], "stride": [1],
                "num_channels": 768, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [25], "stride": [1],
                "num_channels": 768, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
                "residual": True
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [29], "stride": [1],
                "num_channels": 896, "padding": "SAME",
                "dilation":[2], "dropout_keep_prob": 0.6,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [1], "stride": [1],
                "num_channels": 1024, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.6,
            }
        ],

        "dropout_keep_prob": 0.7,

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

        # params for decoding the sequence with language model
        "beam_width": 128,
        "alpha": 2.0,
        "beta": 1.5,

        "decoder_library_path": "ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
        "lm_path":   data_root+"LM/librivox-lm3.binary",
        "trie_path": data_root+"LM/librivox-lm3.trie",
        "alphabet_config_path": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
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
        # "dataset_files": [
        #     data_root + "librivox-dev-clean.csv",
        # ],
        "dataset_files": [
            data_root+"librivox-train-clean-100.csv",
            data_root+"librivox-train-clean-360.csv",
            data_root+"librivox-train-other-500.csv",
            # Add synthetic csv here
        ],
        "syn_enable": False, # Change to True if using synthetic data
        "syn_subdirs": [], # Add subdirs of synthetic data
        "max_duration": 16.7,
        "shuffle": True,
    },
}

eval_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            data_root+"librivox-dev-clean.csv",
        ],
        "shuffle": False,
    },
}

infer_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            data_root+"librivox-test-clean.csv",
        ],
        "shuffle": False,
    },
}
