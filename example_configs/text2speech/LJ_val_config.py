import tensorflow as tf
from open_seq2seq.models import Text2Speech
from open_seq2seq.encoders import Tacotron2Encoder
from open_seq2seq.decoders import Tacotron2Decoder
from open_seq2seq.data import Text2SpeechDataLayer
from open_seq2seq.losses import MeanSquaredErrorLoss, BasicMeanSquaredErrorLoss
from open_seq2seq.optimizers.lr_policies import fixed_lr


base_model = Text2Speech

num_audio_features = 513

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_epochs": 5000,

  "num_gpus": 4,
  # 'gpu_ids': [1],
  "batch_size_per_gpu": 32,

  "save_summaries_steps": 250,
  "print_loss_steps": 25,
  "print_samples_steps": 250,
  "eval_steps": 250,
  "save_checkpoint_steps": 2000,
  "logdir": "result/tacotron-LJ-val",
  "max_grad_norm":1.,

  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": fixed_lr,
  "lr_policy_params": {
    "learning_rate": 1e-3,
  },
  # "lr_policy": exp_decay,
  # "lr_policy_params": {
  #   "begin_decay_at": 0,
  #   "decay_steps": 500,
  #   "decay_rate": 0.9,
  #   "use_staircase_decay": True,
  #   "min_lr": 0.0,
  # },
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
    'src_emb_size': 256,
    "conv_layers": [
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 256, "padding": "SAME"
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 256, "padding": "SAME"
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 256, "padding": "SAME"
      }
    ],
    "activation_fn": tf.nn.relu,
    "enable_bn" : True,

    "num_rnn_layers": 1,
    "rnn_cell_dim": 128,
    "use_cudnn_rnn": False,
    "rnn_type": "lstm",
    "rnn_unidirectional": False,

    "data_format": "channels_last",
  },

  "decoder": Tacotron2Decoder,
  "decoder_params": {
    'attention_layer_size': 128,
    'attention_type': 'location',
    'attention_rnn_enable': True,
    'attention_rnn_units': 512,
    'attention_rnn_layers': 1,
    'attention_rnn_cell_type': 'lstm',

    'decoder_cell_units': 512,
    'decoder_cell_type': 'lstm',
    'decoder_layers': 1,
    'decoder_use_skip_connections': False,
    
    'enable_prenet': True,
    'prenet_layers': 2,
    'prenet_units': 256,

    "anneal_sampling_prob": True,
    'scheduled_sampling_prob': 0.,
    "sampling_test": False,

    'enable_postnet': True,
    "postnet_keep_dropout_prob": 0.5,
    "postnet_data_format": "channels_last",
    "postnet_enable_bn": True,
    "postnet_conv_layers": [
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 256, "padding": "SAME",
        "activation_fn": tf.nn.tanh
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 256, "padding": "SAME",
        "activation_fn": tf.nn.tanh
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 256, "padding": "SAME",
        "activation_fn": tf.nn.tanh
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": num_audio_features, "padding": "SAME",
        "activation_fn": None
      }
    ],
  },
  
  "loss": MeanSquaredErrorLoss,
  "loss_params": {
    "use_mask": True
  },
  # "loss": BasicMeanSquaredErrorLoss,
  # "loss_params": {
  #   "output_key": "post_net_output",
  # },
}

train_params = {
  "data_layer": Text2SpeechDataLayer,
  "data_layer_params": {
    "num_audio_features": num_audio_features,
    "output_type": "spectrogram_disk",
    "vocab_file": "/data/speech/LJSpeech/vocab.txt",
    "dataset_files": [
      "/data/speech/LJSpeech/val_128.csv",
    ],
    'dataset_location':"/data/speech/LJSpeech/wavs/",
    "shuffle": True,
    "mag_power": 2,
    "feature_normalize": False,
  },
}

eval_params = {
  "data_layer": Text2SpeechDataLayer,
  "data_layer_params": {
    "num_audio_features": num_audio_features,
    "output_type": "spectrogram_disk",
    "vocab_file": "/data/speech/LJSpeech/vocab.txt",
    "dataset_files": [
      "/data/speech/LJSpeech/val_128.csv",
    ],
    'dataset_location':"/data/speech/LJSpeech/wavs/",
    "shuffle": False,
    "mag_power": 2,
    "feature_normalize": False,
  },
}