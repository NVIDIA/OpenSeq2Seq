# pylint: skip-file
from open_seq2seq.models import Image2Label
from open_seq2seq.encoders.cnn_encoder import CNNEncoder
from open_seq2seq.decoders import FullyConnectedDecoder
from open_seq2seq.losses import CrossEntropyLoss
from open_seq2seq.data import ImagenetDataLayer
from open_seq2seq.optimizers.lr_policies import poly_decay
import tensorflow as tf


base_model = Image2Label

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_epochs": 120,

  "num_gpus": 4,
  "batch_size_per_gpu": 256,
  "dtype": tf.float32,

  "save_summaries_steps": 2000,
  "print_loss_steps": 100,
  "print_samples_steps": 2000,
  "eval_steps": 5000,
  "save_checkpoint_steps": 5000,
  "logdir": "experiments/alexnet-imagenet",

  "optimizer": "Momentum",
  "optimizer_params": {
    "momentum": 0.90,
  },
  "lr_policy": poly_decay,
  "lr_policy_params": {
    "learning_rate": 0.04,
    "power": 1.0,
  },

  "initializer": tf.variance_scaling_initializer,

  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 0.0005,
  },
  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],
  "encoder": CNNEncoder,
  "encoder_params": {
    'data_format': 'channels_first',
    'cnn_layers': [
      (tf.layers.conv2d, {
        'filters': 64, 'kernel_size': (11, 11),
        'strides': (4, 4), 'padding': 'VALID',
        'activation': tf.nn.relu,
      }),
      (tf.layers.max_pooling2d, {
        'pool_size': (3, 3), 'strides': (2, 2),
      }),
      (tf.layers.conv2d, {
        'filters': 192, 'kernel_size': (5, 5),
        'strides': (1, 1), 'padding': 'SAME',
        'activation': tf.nn.relu,
      }),
      (tf.layers.max_pooling2d, {
        'pool_size': (3, 3), 'strides': (2, 2),
      }),
      (tf.layers.conv2d, {
        'filters': 384, 'kernel_size': (3, 3),
        'strides': (1, 1), 'padding': 'SAME',
        'activation': tf.nn.relu,
      }),
      (tf.layers.conv2d, {
        'filters': 256, 'kernel_size': (3, 3),
        'strides': (1, 1), 'padding': 'SAME',
        'activation': tf.nn.relu,
      }),
      (tf.layers.conv2d, {
        'filters': 256, 'kernel_size': (3, 3),
        'strides': (1, 1), 'padding': 'SAME',
        'activation': tf.nn.relu,
      }),
      (tf.layers.max_pooling2d, {
        'pool_size': (3, 3), 'strides': (2, 2),
      }),
    ],
    'fc_layers': [
      (tf.layers.dense, {'units': 4096, 'activation': tf.nn.relu}),
      (tf.layers.dropout, {'rate': 0.5}),
      (tf.layers.dense, {'units': 4096, 'activation': tf.nn.relu}),
      (tf.layers.dropout, {'rate': 0.5}),
    ],
  },

  "decoder": FullyConnectedDecoder,
  "decoder_params": {
    "output_dim": 1000,
  },
  "loss": CrossEntropyLoss,
  "data_layer": ImagenetDataLayer,
  "data_layer_params": {
    "data_dir": "data/tf-imagenet",
    "image_size": 227,
    "num_classes": 1000,
  },
}