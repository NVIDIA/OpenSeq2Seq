# pylint: skip-file
import os

import tensorflow as tf

from open_seq2seq.data import Text2SpeechDataLayer
from open_seq2seq.decoders import CentaurDecoder
from open_seq2seq.encoders import CentaurEncoder
from open_seq2seq.losses import Text2SpeechLoss
from open_seq2seq.models import Text2SpeechCentaur
from open_seq2seq.optimizers.lr_policies import poly_decay
from open_seq2seq.optimizers.novograd import NovoGrad

base_model = Text2SpeechCentaur

dataset = "LJ"
dataset_location = "/data/LJSpeech"
output_type = "both"

trim = False
exp_mag = True
mag_num_feats = 513
train = "train.csv"
valid = "test.csv"
batch_size = 32
num_audio_features = {
  "mel": 80,
  "magnitude": mag_num_feats
}
data_min = {
  "mel": 1e-2,
  "magnitude": 1e-5,
}

debug = False

num_gpus = 8 if not debug else 1

reduction_factor = 2
attention_layers = 4
encoder_hidden_size = 256
decoder_hidden_size = 512

base_params = {
  "random_seed": 0,
  "use_horovod": True if not debug else False,
  "max_steps": 1000000,
  "bench_start": 0,

  "num_gpus": num_gpus,
  "batch_size_per_gpu": batch_size,

  "save_summaries_steps": 1000 if not debug else 10,
  "print_loss_steps": 1000 if not debug else 10,
  "print_samples_steps": 1000 if not debug else 10,
  "eval_steps": 5000 if not debug else 50,
  "save_checkpoint_steps": 5000,
  "save_to_tensorboard": True,
  "logdir": "result/centaur-float",
  "max_grad_norm": 1.,

  "optimizer": NovoGrad,
  "optimizer_params": {
    "beta1": 0.95,
    "beta2": 0.98,
    "epsilon": 1e-08,
    "weight_decay": 0.001,

  },
  "lr_policy": poly_decay,
  "lr_policy_params": {
    "learning_rate": 0.02,
    "power": 2.0,
  },
  "dtype": tf.float32,
  "initializer": tf.contrib.layers.xavier_initializer,

  "summaries": ["learning_rate", "variables", "gradients", "larc_summaries",
                "variable_norm", "gradient_norm", "global_gradient_norm"],

  "encoder": CentaurEncoder,
  "encoder_params": {
    "src_vocab_size": 94,
    "embedding_size": encoder_hidden_size,
    "output_size": encoder_hidden_size,
    "pad_embeddings_2_eight": True,
    "cnn_dropout_prob": 0.1,
    "conv_layers": [
      {
        "kernel_size": [3], "stride": [1],
        "num_channels": encoder_hidden_size, "padding": "SAME",
        "activation_fn": tf.nn.relu
      },
      {
        "kernel_size": [3], "stride": [1],
        "num_channels": encoder_hidden_size, "padding": "SAME",
        "activation_fn": tf.nn.relu
      },
      {
        "kernel_size": [3], "stride": [1],
        "num_channels": encoder_hidden_size, "padding": "SAME",
        "activation_fn": tf.nn.relu
      },
      {
        "kernel_size": [3], "stride": [1],
        "num_channels": encoder_hidden_size, "padding": "SAME",
        "activation_fn": tf.nn.relu
      }
    ]
  },

  "decoder": CentaurDecoder,
  "decoder_params": {
    "attention_layers": attention_layers,
    "self_attention_conv_params": {
      "kernel_size": [5],
      "stride": [1],
      "num_channels": decoder_hidden_size,
      "padding": "VALID",
      "is_causal": True,
      "activation_fn": tf.nn.relu
    },

    "window_size": 4,
    "back_step_size": 0,
    "force_layers": [1, 3],

    "hidden_size": decoder_hidden_size,
    "reduction_factor": reduction_factor,
    "prenet_layers": 2,
    "prenet_hidden_size": decoder_hidden_size,
    "prenet_use_inference_dropout": False,
    "cnn_dropout_prob": 0.1,
    "prenet_dropout": 0.5,
    "conv_layers":
      [
        {
          "kernel_size": [5],
          "stride": [1],
          "num_channels": decoder_hidden_size,
          "padding": "VALID",
          "is_causal": True,
          "activation_fn": tf.nn.relu
        }
      ] * 4,
    "mag_conv_layers":
      [
        {
          "kernel_size": [5],
          "stride": [1],
          "num_channels": decoder_hidden_size,
          "padding": "VALID",
          "is_causal": True,
          "activation_fn": tf.nn.relu
        }
      ] * 4,
    "attention_dropout": 0.1,
    "layer_postprocess_dropout": 0.1
  },

  "loss": Text2SpeechLoss,
  "loss_params": {
    "use_mask": True,
    "l1_norm": True
  },

  "data_layer": Text2SpeechDataLayer,
  "data_layer_params": {
    "dataset": dataset,
    "use_cache": True,
    "num_audio_features": num_audio_features,
    "output_type": output_type,
    "vocab_file": "open_seq2seq/test_utils/vocab_tts.txt",
    "dataset_location": dataset_location,
    "mag_power": 1,
    "pad_EOS": True,
    "feature_normalize": False,
    "feature_normalize_mean": 0.,
    "feature_normalize_std": 1.,
    "data_min": data_min,
    "mel_type": "htk",
    "trim": trim,
    "duration_max": 1024,
    "duration_min": 24,
    "exp_mag": exp_mag
  },
}

train_params = {
  "data_layer_params": {
    "dataset_files": [
      os.path.join(dataset_location, train),
    ],
    "shuffle": True,
  },
}

eval_params = {
  "data_layer_params": {
    "dataset_files": [
      os.path.join(dataset_location, valid),
    ],
    "duration_max": 1000,
    "duration_min": 0,
    "shuffle": False,
  },
}

infer_params = {
  "data_layer_params": {
    "dataset_files": [
      os.path.join(dataset_location, "infer.csv"),
    ],
    "duration_max": 1000,
    "duration_min": 0,
    "shuffle": False,
  },
}

interactive_infer_params = {
  "data_layer_params": {
    "dataset_files": [],
    "duration_max": 1000,
    "duration_min": 0,
    "shuffle": False,
  },
}
