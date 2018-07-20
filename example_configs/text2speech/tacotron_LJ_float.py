# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Text2Speech
from open_seq2seq.encoders import Tacotron2Encoder
from open_seq2seq.decoders import Tacotron2Decoder
from open_seq2seq.data import Text2SpeechDataLayer
from open_seq2seq.losses import TacotronLoss
from open_seq2seq.optimizers.lr_policies import fixed_lr, transformer_policy, exp_decay


base_model = Text2Speech

output_type = "magnitude"

if output_type == "magnitude":
  num_audio_features = 513
elif output_type == "mel":
  num_audio_features = 80

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "max_steps": 100000,

  "num_gpus": 1,
  "batch_size_per_gpu": 32,

  "save_summaries_steps": 50,
  "print_loss_steps": 50,
  "print_samples_steps": 500,
  "eval_steps": 500,
  "save_checkpoint_steps": 2500,
  "save_to_tensorboard": True,
  "logdir": "result/tacotron-LJ-float",
  "max_grad_norm":1.,

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
  "loss_scaling": 1.,
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
  },

  "decoder": Tacotron2Decoder,
  "decoder_params": {
    "zoneout_prob": 0.1,
    
    'attention_type': 'location',
    'attention_bias': True,

    'decoder_cell_units': 1024,
    'decoder_cell_type': tf.nn.rnn_cell.LSTMCell,
    'decoder_layers': 2,
    
    'enable_prenet': True,
    'prenet_layers': 2,
    'prenet_units': 256,

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
        "num_channels": num_audio_features, "padding": "SAME",
        "activation_fn": None
      }
    ],
    "mask_decoder_sequence": True,
    "parallel_iterations": 32,
  },
  
  "loss": TacotronLoss,
  "loss_params": {
    "use_mask": True
  },

  "data_layer": Text2SpeechDataLayer,
  "data_layer_params": {
    "num_audio_features": num_audio_features,
    "output_type": output_type,
    "vocab_file": "/data/speech/LJSpeech/vocab_EOS_80.txt",
    'dataset_location':"/data/speech/LJSpeech/wavs/",
    "mag_power": 1,
    "pad_EOS": True,
    "feature_normalize": False,
    "feature_normalize_mean": 0.,
    "feature_normalize_std": 1.,
  },
}

train_params = {
  "data_layer_params": {
    "dataset_files": [
      "/data/speech/LJSpeech/train.csv",
    ],
    "shuffle": True,
  },
}

eval_params = {
  "data_layer_params": {
    "dataset_files": [
      "/data/speech/LJSpeech/new_val.csv",
    ],
    "shuffle": False,
  },
}

infer_params = {
  "data_layer_params": {
    "dataset_files": [
      "/data/speech/LJSpeech/test.csv",
    ],
    "shuffle": False,
  },
}
