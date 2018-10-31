# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Text2SpeechWavenet
from open_seq2seq.encoders import WavenetEncoder
from open_seq2seq.decoders import FakeDecoder
from open_seq2seq.losses import WavenetLoss
from open_seq2seq.data import WavenetDataLayer
from open_seq2seq.optimizers.lr_policies import exp_decay
from open_seq2seq.parts.convs2s.utils import gated_linear_units

base_model = Text2SpeechWavenet

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "max_steps": 1000000,

  "num_gpus": 1,
  "batch_size_per_gpu": 4,

  "save_summaries_steps": 50,
  "print_loss_steps": 50,
  "print_samples_steps": 500,
  "eval_steps": 500,
  "save_checkpoint_steps": 2500,
  "logdir": "result/wavenet-LJ-mixed",

  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": exp_decay,
  "lr_policy_params": {
    "learning_rate": 1e-3,
    "decay_steps": 20000,
    "decay_rate": 0.1,
    "use_staircase_decay": False,
    "begin_decay_at": 45000,
    "min_lr": 1e-5,
  },
  "dtype": "mixed",
  "loss_scaling": "Backoff",
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    "scale": 1e-6
  },
  "initializer": tf.contrib.layers.xavier_initializer,

  "summaries": [],

  "encoder": WavenetEncoder,
  "encoder_params": {
    "layer_type": "conv1d",
    "kernel_size": 3,
    "strides": 1,
    "padding": "VALID",
    "blocks": 3,
    "layers_per_block": 10,
    "filters": 64,
    "quantization_channels": 256
  },

  "decoder": FakeDecoder,

  "loss": WavenetLoss,

  "data_layer": WavenetDataLayer,
  "data_layer_params": {
    "num_audio_features": 80,
    "dataset_location": "data/speech/LJSpeech/wavs/"
  }
}

train_params = {
  "data_layer_params": {
    "dataset_files": [
      "data/speech/LJSpeech/train.csv",
    ],
    "shuffle": True,
  },
}

eval_params = {
  "data_layer_params": {
    "dataset_files": [
      "data/speech/LJSpeech/val.csv",
    ],
    "shuffle": False,
  },
}

infer_params = {
  "data_layer_params": {
    "dataset_files": [
      "data/speech/LJSpeech/test.csv",
    ],
    "shuffle": False,
  },
}

interactive_infer_params = {
  "data_layer_params": {
    "dataset_files": [],
    "shuffle": False,
  },
}
