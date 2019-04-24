# Copyright (c) 2019 NVIDIA Corporation
import tensorflow as tf


class Prenet:
  """
  Centaur decoder pre-net.
  """

  def __init__(self,
               n_layers,
               hidden_size,
               activation_fn,
               dropout=0.5,
               regularizer=None,
               training=True,
               dtype=None,
               name="prenet"):
    """
    Pre-net constructor.

    Args:
      n_layers: number of fully-connected layers to use.
      hidden_size: number of units in each pre-net layer.
      activation_fn: activation function to use.
      dropout: dropout rate. Defaults to 0.5.
      regularizer: regularizer for the convolution kernel.
        Defaults to None.
      training: whether it is training mode. Defaults to None.
      dtype: dtype of the layer's weights. Defaults to None.
      name: name of the block.
    """

    self.name = name
    self.layers = []
    self.dropout = dropout
    self.training = training

    for i in range(n_layers):
      layer = tf.layers.Dense(
          name="layer_%d" % i,
          units=hidden_size,
          use_bias=True,
          activation=activation_fn,
          kernel_regularizer=regularizer,
          dtype=dtype
      )
      self.layers.append(layer)

  def __call__(self, x):
    with tf.variable_scope(self.name):
      for layer in self.layers:
        x = tf.layers.dropout(
            layer(x),
            rate=self.dropout,
            training=self.training
        )

      return x
