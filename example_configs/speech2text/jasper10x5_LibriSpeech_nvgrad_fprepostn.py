# pylint: skip-file
import tensorflow as tf
import numpy as np
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import TDNNEncoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.data.speech2text.speech2text import Speech2TextDataLayer
from open_seq2seq.losses import CTCLoss
from open_seq2seq.optimizers.lr_policies import poly_decay
from open_seq2seq.optimizers.novograd import NovoGrad

residual_dense = True # Enable or disable Dense Residual

base_model = Speech2Text

base_params = {
    "random_seed": 0,
    "use_horovod": True,
    "num_epochs": 400,

    "num_gpus": 8,
    "batch_size_per_gpu": 32,
    "iter_size": 1,

    "save_summaries_steps": 100,
    "print_loss_steps": 10,
    "print_samples_steps": 2200,
    "eval_steps": 2200,
    "save_checkpoint_steps": 1100,
    "logdir": "jasper_log_folder",
    "num_checkpoints": 2,

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
        "learning_rate": 0.02,
        "min_lr": 1e-5,
        "power": 2.0,
    },
    "larc_params": {
        "larc_eta": 0.001,
    },

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
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [11], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [13], "stride": [1],
                "num_channels": 384, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [13], "stride": [1],
                "num_channels": 384, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [17], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [17], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [21], "stride": [1],
                "num_channels": 640, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [21], "stride": [1],
                "num_channels": 640, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [25], "stride": [1],
                "num_channels": 768, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
                "residual": True, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 5,
                "kernel_size": [25], "stride": [1],
                "num_channels": 768, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
                "residual": True, "residual_dense": residual_dense
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
        "dither": 1e-5,
        "backend": "librosa",
        "gain": 6.157502732847814e-05,
        "features_mean": 
          np.array([-12.34298268, -10.93171946, -10.43775974,  -9.67980806,
                     -9.69266087,  -9.49124337,  -9.62320113,  -9.31896693,
                     -9.33991249,  -9.17555551,  -9.32361146,  -9.38353122,
                     -9.6434601 ,  -9.8221681 , -10.03372666, -10.15247738,
                    -10.23399336, -10.32032843, -10.34099102, -10.43911045,
                    -10.41168591, -10.47855113, -10.51948346, -10.42248584,
                    -10.48503355, -10.40489218, -10.27495663, -10.10816251,
                    -10.03621187,  -9.92948895,  -9.89640601,  -9.79033564,
                     -9.77823043,  -9.74417924,  -9.76694271,  -9.77316719,
                     -9.76335741,  -9.75990194,  -9.67838081,  -9.61090049,
                     -9.53484618,  -9.5191552 ,  -9.54069545,  -9.53790944,
                     -9.55083911,  -9.59862674,  -9.64561474,  -9.69590462,
                     -9.72453847,  -9.76581933,  -9.82116916, -10.00280741,
                    -10.16394213, -10.33968281, -10.50684207, -10.6788766 ,
                    -10.81749469, -10.93601949, -11.00201463, -11.06010048,
                    -11.14923441, -11.26092026, -11.43304507, -11.61147154]),
        "features_std_dev":
          np.array([2.61237426, 3.33758081, 4.10880692, 4.50130923, 4.58097235,
                    4.52404595, 4.53129841, 4.5977929 , 4.63095334, 4.60578638,
                    4.55117136, 4.48328155, 4.42233164, 4.34985156, 4.28006312,
                    4.19936861, 4.12378264, 4.0459167 , 3.98277463, 3.92930132,
                    3.88786137, 3.85849594, 3.84815856, 3.83083361, 3.81054102,
                    3.78336743, 3.68238327, 3.57526528, 3.6195817 , 3.73389776,
                    3.74648978, 3.73240959, 3.71407061, 3.68744318, 3.65515504,
                    3.61819041, 3.60997782, 3.61777861, 3.63578848, 3.66737302,
                    3.68144978, 3.66504601, 3.63489252, 3.60950061, 3.59533474,
                    3.57621649, 3.54884579, 3.5353986 , 3.53084464, 3.50953945,
                    3.4406293 , 3.45784911, 3.4502744 , 3.42613901, 3.38972961,
                    3.37488988, 3.35147518, 3.32241393, 3.27641687, 3.24996724,
                    3.20865607, 3.14300007, 3.11081261, 3.05606053])
    },
}

train_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "augmentation": {
            'speed_perturbation_ratio': [0.9, 1., 1.1],
        },
        "dataset_files": [
            "/data/librispeech/librivox-train-clean-100.csv",
            "/data/librispeech/librivox-train-clean-360.csv",
            "/data/librispeech/librivox-train-other-500.csv"
        ],
        "max_duration": 16.7,
        "shuffle": True,
    },
}

eval_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "dataset_files": [
            "/data/librispeech/librivox-dev-clean.csv",
        ],
        "shuffle": False,
    },
}

infer_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "dataset_files": [
            "/data/librispeech/librivox-test-clean.csv",
        ],
        "shuffle": False,
    },
}
