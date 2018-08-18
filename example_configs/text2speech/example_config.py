# pylint: skip-file
import os
import tensorflow as tf
from open_seq2seq.models import Text2Speech
from open_seq2seq.encoders import Tacotron2Encoder
from open_seq2seq.decoders import Tacotron2Decoder
from open_seq2seq.data import Text2SpeechDataLayer
from open_seq2seq.losses import TacotronLoss
from open_seq2seq.optimizers.lr_policies import fixed_lr, transformer_policy, exp_decay


base_model = Text2Speech

data_root = replace

# data_root = "/raid2/MAILABS/de_DE/by_book/female/eva_k/toten_seelen/"
# data_root = "/raid2/MAILABS/en_US/by_book/female/mary_ann/northandsouth/"
# data_root = "/data/speech/LJSpeech/"
# data_root = "/data/librispeech/"

output_type = "magnitude"
style_mode = None

if style_mode == None:
  style_enable = False
  add_kl = False
elif style_mode == "vae":
  style_enable = True
  add_kl = True
elif style_mode == "attention":
  style_enable = True
  add_kl = False
else:
  raise ValueError("Unknown style mode")

append = ""
if "by_book" in data_root:
  trim = True
  dataset = "MAILABS-16"
  mag_num_feats = 401
elif "LJSpeech" in data_root:
  trim = False
  dataset = "LJ"
  mag_num_feats = 513
  append = "_32"
elif "librispeech" in data_root.lower():
  trim = False
  dataset = "Librispeech"
  mag_num_feats = 401
else:
  raise ValueError("Unknown dataset")

if output_type == "magnitude":
  num_audio_features = mag_num_feats
  data_min = 1e-5
  output_type = "magnitude_disk"
elif output_type == "mel":
  num_audio_features = 80
  data_min = 1e-2
  output_type = "mel_disk"
elif output_type == "both":
  num_audio_features = {
      "mel": 80,
      "magnitude": mag_num_feats
  }
  data_min = {
      "mel": 1e-2,
      "magnitude": 1e-5,
  }
  output_type = "both_disk"
else:
  raise ValueError("Unknown param for output_type")


base_params = {
  "random_seed": 0,
  "use_horovod": False,
  # "num_epochs": 501,
  "max_steps": 100000,

  "num_gpus": 1,
  # 'gpu_ids': [1],
  "batch_size_per_gpu": 32,

  "save_summaries_steps": 50,
  "print_loss_steps": 50,
  "print_samples_steps": 500,
  "eval_steps": 500,
  "save_checkpoint_steps": 2500,
  "save_to_tensorboard": False,
  "logdir": "result/tacotron-float",
  "max_grad_norm":1.,
  # "larc_params": {
  #   "larc_eta": 0.001,
  # },

  "optimizer": "Adam",
  "optimizer_params": {},
  # "lr_policy": fixed_lr,
  # "lr_policy_params": {
  #   "learning_rate": 1e-3,
  # },
  # "lr_policy": transformer_policy,
  # "lr_policy_params": {
  #   "learning_rate": 1.8,
  #   "max_lr": 1e-3,
  #   "warmup_steps": 5000,
  #   "d_model": 64,
  #   "coefficient": 1
  # },
  "lr_policy": exp_decay,
  "lr_policy_params": {
    "learning_rate": 1e-3,
    "decay_steps": 20000,
    "decay_rate": 0.1,
    "use_staircase_decay": False,
    "begin_decay_at": 45000,
    "min_lr": 1e-5,
  },
  # "dtype": tf.float32, "mixed", tf.float16
  "dtype": tf.float32,
  # weight decay
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 1e-6
  },
  "initializer": tf.contrib.layers.xavier_initializer,

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],

  "encoder": Tacotron2Encoder,
  "encoder_params": {
    "dropout_keep_prob": 0.5,
    'src_emb_size': 512,
    "conv_layers": [
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME"
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME"
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME"
      }
    ],
    "activation_fn": tf.nn.relu,

    "num_rnn_layers": 1,
    "rnn_cell_dim": 256,
    "rnn_unidirectional": False,
    # "use_cudnn_rnn": False,
    # "rnn_type": tf.nn.rnn_cell.LSTMCell,
    # "zoneout_prob": 0.1,
    "use_cudnn_rnn": True,
    "rnn_type": tf.contrib.cudnn_rnn.CudnnLSTM,
    "zoneout_prob": 0.,

    "data_format": "channels_last",

    "style_embedding_enable": style_enable,
    "style_embedding_params": {
      "conv_layers": [
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 32, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 32, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 64, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 64, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 128, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 128, "padding": "SAME"
        }
      ],
      "num_rnn_layers": 1,
      "rnn_cell_dim": 128,
      "rnn_unidirectional": False,
      "rnn_type": tf.nn.rnn_cell.LSTMCell,
      "num_tokens": 10,
      "mode": style_mode,
      "emb_size": 256
    }
  },

  "decoder": Tacotron2Decoder,
  "decoder_params": {
    "zoneout_prob": 0.,
    "dropout_prob": 0.1,
    
    'attention_layer_size': 128,
    'attention_type': 'location',
    'attention_rnn_enable': False,
    'attention_rnn_units': 1024,
    'attention_rnn_layers': 1,
    'attention_rnn_cell_type': tf.nn.rnn_cell.LSTMCell,
    'attention_bias': True,
    'use_state_for_location': True,

    'decoder_cell_units': 1024,
    'decoder_cell_type': tf.nn.rnn_cell.LSTMCell,
    'decoder_layers': 2,
    
    'enable_prenet': True,
    'prenet_layers': 2,
    'prenet_units': 256,

    "anneal_teacher_forcing": False,
    "anneal_teacher_forcing_stop_gradient": False,
    'scheduled_sampling_prob': 0.,

    'enable_postnet': True,
    "postnet_keep_dropout_prob": 0.5,
    "postnet_data_format": "channels_last",
    "postnet_conv_layers": [
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME",
        "activation_fn": tf.nn.tanh
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME",
        "activation_fn": tf.nn.tanh
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME",
        "activation_fn": tf.nn.tanh
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME",
        "activation_fn": tf.nn.tanh
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": -1, "padding": "SAME",
        "activation_fn": None
      }
    ],
    "mask_decoder_sequence": True,
    "parallel_iterations": 32,
    "stop_token_choice": 2,
  },
  
  "loss": TacotronLoss,
  "loss_params": {
    "use_mask": True,
    "add_kl": add_kl
  },

  "data_layer": Text2SpeechDataLayer,
  "data_layer_params": {
    "dataset": dataset,
    "num_audio_features": num_audio_features,
    "output_type": output_type,
    "vocab_file": "open_seq2seq/test_utils/vocab_tts.txt",
    'dataset_location': os.path.join(data_root, "wavs"),
    "mag_power": 1,
    "pad_EOS": True,
    "feature_normalize": False,
    "feature_normalize_mean": 0.,
    "feature_normalize_std": 1.,
    "trim": trim,
    "duration_max":875,
    "duration_min":100,
    "data_min":data_min,
    "mel_type":'htk',
  },
}

train_params = {
  "data_layer_params": {
    "dataset_files": [
      os.path.join(data_root, "train{}.csv".format(append)),
    ],
    "shuffle": True,
  },
}

eval_params = {
  "data_layer_params": {
    "dataset_files": [
      os.path.join(data_root, "val{}.csv".format(append)),
    ],
    "shuffle": False,
  },
}

infer_params = {
  "data_layer_params": {
    "dataset_files": [
      os.path.join(data_root, "test.csv"),
    ],
    "shuffle": False,
  },
}