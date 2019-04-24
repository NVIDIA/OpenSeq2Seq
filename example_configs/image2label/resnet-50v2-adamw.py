# pylint: skip-file
from open_seq2seq.models import Image2Label
from open_seq2seq.encoders import ResNetEncoder
from open_seq2seq.decoders import FullyConnectedDecoder
from open_seq2seq.losses import CrossEntropyLoss
from open_seq2seq.data import ImagenetDataLayer

import tensorflow as tf

data_root = ""

base_model = Image2Label

base_params = {
  "random_seed": 0,
  "use_horovod":  False, #True,
  "num_gpus": 8,
  "batch_size_per_gpu": 128,

  "num_epochs": 100,

  "dtype": "mixed",
  "loss_scaling": "Backoff",

  "save_summaries_steps": 2000,
  "print_loss_steps": 100,
  "print_samples_steps": 10000,
  "eval_steps": 5000,
  "save_checkpoint_steps": 50000,
  "logdir": "logs/rn50-adamw",

  "optimizer": "AdamW",
  "optimizer_params": {
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-08,
    "weight_decay": 0.1,
  },

  "lr_policy": tf.train.cosine_decay,
  "lr_policy_params": {
    "learning_rate": 0.002, # 8 GPUs
  },

  "initializer": tf.variance_scaling_initializer,

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],
  "encoder": ResNetEncoder,
  "encoder_params": {
    'resnet_size': 50,
    "regularize_bn": False,
  },
  "decoder": FullyConnectedDecoder,
  "decoder_params": {
    "output_dim": 1000,
  },
  "loss": CrossEntropyLoss,
  "data_layer": ImagenetDataLayer,
  "data_layer_params": {
    "data_dir": data_root+"data", # "data",
    "image_size": 224,
    "num_classes": 1000,
  },
}
