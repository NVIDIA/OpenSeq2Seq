from open_seq2seq.models import Image2Label
from open_seq2seq.encoders import ResNetEncoder
from open_seq2seq.decoders import FullyConnectedDecoder
from open_seq2seq.losses import CrossEntropyLoss
from open_seq2seq.data import ImagenetDataLayer
from open_seq2seq.optimizers.lr_policies import poly_decay2
import tensorflow as tf


base_model = Image2Label

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_epochs": 25,

  "num_gpus": 8,
  "batch_size_per_gpu": 32,
  "dtype": tf.float32,

  "save_summaries_steps": 2000,
  "print_loss_steps": 100,
  "print_samples_steps": 2000,
  "eval_steps": 5000,
  "save_checkpoint_steps": 5000,
  "logdir": "experiments/resnet50-imagenet",

  "optimizer": "Momentum",
  "optimizer_params": {
    "momentum": 0.90,
  },
  "lr_policy": poly_decay2,
  "lr_policy_params": {
    "learning_rate": 1.0,
    "power": 2.0,
  },
  "larc_params": {
    "larc_eta": 0.001,
  },

  "initializer": tf.variance_scaling_initializer,

  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 0.0001,
  },
  "increase_regularization_from_zero": False,
  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],
  "encoder": ResNetEncoder,
  "encoder_params": {
    'resnet_size': 50,
    "regularize_bn": False,
  },
  "decoder": FullyConnectedDecoder,
  "decoder_params": {
    "output_dim": 1001,
  },
  "loss": CrossEntropyLoss,
  "loss_params": {
    "increase_smoothing_from_zero": False,
    "label_smoothing": 0.1,
  },
  "data_layer": ImagenetDataLayer,
  "data_layer_params": {
    "data_dir": "data/tf-imagenet",
  },
}
