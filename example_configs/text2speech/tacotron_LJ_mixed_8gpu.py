# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Text2Speech
from open_seq2seq.encoders import Tacotron2Encoder
from open_seq2seq.decoders import Tacotron2Decoder
from open_seq2seq.data import Text2SpeechDataLayer
from open_seq2seq.losses import TacotronLoss
from open_seq2seq.optimizers.lr_policies import fixed_lr, transformer_policy, exp_decay


base_model = Text2Speech

output_type = "mel"

if output_type == "magnitude":
  num_audio_features = 513
  data_min = 1e-5
elif output_type == "mel":
  num_audio_features = 80
  data_min = 1e-2

base_params = {
  "random_seed": 0,
  "use_horovod": True,
  "max_steps": 40000,

  "batch_size_per_gpu": 48,

  "save_summaries_steps": 50,
  "print_loss_steps": 50,
  "print_samples_steps": 500,
  "eval_steps": 500,
  "save_checkpoint_steps": 2500,
  "save_to_tensorboard": True,
  "logdir": "result/tacotron-LJ-mixed",
  "larc_params": {
    "larc_eta": 0.001,
  },

  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": exp_decay,
  "lr_policy_params": {
    "learning_rate": 1e-3,
    "decay_steps": 10000,
    "decay_rate": 0.1,
    "use_staircase_decay": False,
    "begin_decay_at": 20000,
    "min_lr": 1e-5,
  },
  "dtype": "mixed",
  "loss_scaling": "Backoff",
  "loss_scaling_params": {
    "scale_min": 1.,
    "scale_max": 65536.,
  },
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 1e-6
  },
  "initializer": tf.contrib.layers.xavier_initializer,

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm',
                'loss_scale'],

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
    "use_cudnn_rnn": True,
    "rnn_type": tf.contrib.cudnn_rnn.CudnnLSTM,
    "zoneout_prob": 0.,

    "data_format": "channels_last",
  },

  "decoder": Tacotron2Decoder,
  "decoder_params": {
    "zoneout_prob": 0.,
    "dropout_prob": 0.1,
    
    'attention_type': 'location',
    'attention_layer_size': 128,
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
    "dataset": "LJ",
    "num_audio_features": num_audio_features,
    "output_type": output_type,
    "vocab_file": "open_seq2seq/test_utils/vocab_tts.txt",
    'dataset_location':"/data/speech/LJSpeech/wavs/",
    "mag_power": 1,
    "pad_EOS": True,
    "feature_normalize": False,
    "feature_normalize_mean": 0.,
    "feature_normalize_std": 1.,
    "data_min":data_min,
    "mel_type":'htk',
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
      "/data/speech/LJSpeech/val.csv",
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
