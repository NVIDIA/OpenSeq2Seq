# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import TSSEncoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.data.speech2text.speech2text import Speech2TextDataLayer
from open_seq2seq.losses import CTCLoss
from open_seq2seq.optimizers.lr_policies import poly_decay
from open_seq2seq.optimizers.novograd import NovoGrad

base_model = Speech2Text
data_root = '/data/librispeech/'
d_model = 1024
residual = True
residual_dense = True
repeat_1 = 4
repeat_2 = 4
dropout_factor = 1.

base_params = {
    "random_seed": 0,
    "use_horovod": True,
    "num_epochs": 50,

    "num_gpus": 1,
    "batch_size_per_gpu": 32,
    "iter_size": 1,

    "save_summaries_steps": 100,
    "print_loss_steps": 10,
    "print_samples_steps": 2200,
    "eval_steps": 2200,
    "save_checkpoint_steps": 1100,
    "num_checkpoints": 5,
    "logdir": "yasper_log_folder",

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
      "learning_rate": 0.05,
      "power": 2.0,
    },

    "regularizer": tf.contrib.layers.l2_regularizer,
    "regularizer_params": {
        'scale': 0.001
    },

    "dtype": "mixed",
    "loss_scaling": "Backoff",

    "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                  'variable_norm', 'gradient_norm', 'global_gradient_norm'],

    "encoder": TSSEncoder,
    "encoder_params": {
        "convnet_layers": [
          {
            "type": "conv1d", "repeat": 1,
            "kernel_size": [11], "stride": [2],
            "num_channels": 256, "padding": "SAME",
            "dilation": [1], "dropout_keep_prob": 0.8 * dropout_factor,
          },
          {
            "type": "conv1d", "repeat": repeat_1,
            "kernel_size": [11], "stride": [1],
            "num_channels": 256, "padding": "SAME",
            "dilation": [1], "dropout_keep_prob": 0.8 * dropout_factor,
            "residual": residual, "residual_dense": residual_dense
          },
        ],
        "dropout_keep_prob": 0.7,
        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            'uniform': False,
        },
        "normalization": "batch_norm",
        "activation_fn": lambda x: tf.minimum(tf.nn.relu(x), 20.0),
        "data_format": "channels_last",

        "encoder_layers": [(32, 32)]*12,
        "hidden_size": d_model,
        "num_heads": 16,
        "attention_dropout": 0.1,
        "filter_size": 4 * d_model,
        "relu_dropout": 0.3,
        "layer_postprocess_dropout": 0.3,
    },

    "decoder": FullyConnectedCTCDecoder,
    "decoder_params": {
        "initializer": tf.contrib.layers.xavier_initializer,
        "use_language_model": False,

        # params for decoding the sequence with language model
        "beam_width": 512,
        "alpha": 2.0,
        "beta": 1.5,

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
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            data_root + "librivox-train-clean-100.csv",
            data_root + "librivox-train-clean-360.csv",
            data_root + "librivox-train-other-500.csv",
        ],
        #"max_duration": 8.00,
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
            data_root + "librivox-dev-clean.csv",
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
            data_root + "librivox-test-clean.csv",
        ],
        "shuffle": False,
    },
}
