# pylint: skip-file
from open_seq2seq.models import Image2Label
from open_seq2seq.encoders.cnn_encoder import CNNEncoder
from open_seq2seq.decoders import FullyConnectedDecoder
from open_seq2seq.losses import CrossEntropyLoss
from open_seq2seq.data.image2label.image2label import CifarDataLayer
from open_seq2seq.optimizers.lr_policies import poly_decay
import tensorflow as tf


base_model = Image2Label

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_epochs": 200,

  "num_gpus": 1,
  "batch_size_per_gpu": 32,
  "dtype": tf.float32,

  "save_summaries_steps": 2000,
  "print_loss_steps": 100,
  "print_samples_steps": 2000,
  "eval_steps": 5000,
  "save_checkpoint_steps": 5000,
  "logdir": "experiments/test-cifar",

  "optimizer": "Momentum",
  "optimizer_params": {
    "momentum": 0.90,
  },
  "lr_policy": poly_decay,
  "lr_policy_params": {
    "learning_rate": 0.001,
    "power": 1.0,
  },

  "initializer": tf.variance_scaling_initializer,

  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 0.0002,
  },
  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],
  "encoder": CNNEncoder,
  "encoder_params": {
    'data_format': 'channels_first',
    'cnn_layers': [
      # block 1
      (tf.layers.conv2d, {
        'filters': 128, 'kernel_size': (3, 3),
        'strides': (1, 1), 'padding': 'SAME',
        'activation': tf.nn.relu,
      }),
      (tf.layers.conv2d, {
        'filters': 128, 'kernel_size': (3, 3),
        'strides': (1, 1), 'padding': 'SAME',
        'activation': tf.nn.relu,
      }),
      (tf.layers.conv2d, {
        'filters': 128, 'kernel_size': (3, 3),
        'strides': (1, 1), 'padding': 'SAME',
        'activation': None, 'use_bias': False,
      }),
      (tf.layers.batch_normalization, {'momentum': 0.9, 'epsilon': 0.0001}),
      (tf.nn.relu, {}),
      (tf.layers.max_pooling2d, {
        'pool_size': 3, 'strides': 2, 'padding': 'SAME',
      }),
      # block 2
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
      (tf.layers.conv2d, {
        'filters': 256, 'kernel_size': (3, 3),
        'strides': (1, 1), 'padding': 'SAME',
        'activation': None, 'use_bias': False,
      }),
      (tf.layers.batch_normalization, {'momentum': 0.9, 'epsilon': 0.0001}),
      (tf.nn.relu, {}),
      (tf.layers.max_pooling2d, {
        'pool_size': 3, 'strides': 2, 'padding': 'SAME',
      }),
      # block 3
      (tf.layers.conv2d, {
        'filters': 320, 'kernel_size': (3, 3),
        'strides': (1, 1), 'padding': 'SAME',
        'activation': tf.nn.relu,
      }),
      (tf.layers.conv2d, {
        'filters': 320, 'kernel_size': (1, 1),
        'strides': (1, 1), 'padding': 'SAME',
        'activation': tf.nn.relu,
      }),
    ],
  },

  "decoder": FullyConnectedDecoder,
  "decoder_params": {
    "output_dim": 10,
  },
  "loss": CrossEntropyLoss,
  "data_layer": CifarDataLayer,
  "data_layer_params": {
    "data_dir": "data/cifar-10-batches-bin",
  },
}
