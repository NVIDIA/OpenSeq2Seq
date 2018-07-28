"""Implementation of fully connected network with weight normalization.
Inspired from https://github.com/tobyyouup/conv_seq2seq"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import math


class FeedFowardNetworkNormalized(tf.layers.Layer):
  """Fully connected feedforward network with weight normalization"""

  def __init__(self,
               in_dim,
               out_dim,
               dropout,
               var_scope_name,
               mode,
               normalization_type="weight_norm"):
    """initializes the linear layer.
    This layer projects from in_dim-dimenstional space to out_dim-dimentional space.
    It uses weight normalization (Salimans & Kingma, 2016)  w = g * v/2-norm(v)

    Args:
      in_dim: int last dimension of the inputs
      out_dim: int new dimension for the output
      dropout: float the keep-dropout value used in the previous layer.
                  It is used to initialize the weights. Give 1.0 if no dropout.
      var_scope_name: str the scope name for the weight variables
      mode: str current mode
      normalization_type: str specifies the normalization used for this layer.
                          "weight_norm" for weight normalization or
                          "batch_norm" for batch normalization
    """
    super(FeedFowardNetworkNormalized, self).__init__()
    self.out_dim = out_dim
    self.in_dim = in_dim
    self.normalization_type = normalization_type
    self.var_scope_name = var_scope_name
    self.mode = mode

    if normalization_type == "batch_norm":
      self.apply_batch_norm = True
      self.bias_enabled = False
      self.wn_enabled = False
      self.apply_layer_norm = False
    elif normalization_type == "weight_norm":
      self.apply_batch_norm = False
      self.bias_enabled = True
      self.wn_enabled = True
      self.apply_layer_norm = False
    elif normalization_type == "layer_norm":
      self.apply_batch_norm = False
      self.bias_enabled = True
      self.wn_enabled = True
      self.apply_layer_norm = True
    elif normalization_type is None:
      self.apply_batch_norm = False
      self.bias_enabled = True
      self.wn_enabled = False
      self.apply_layer_norm = False
    else:
      raise ValueError("Wrong normalization type: {}".format(normalization_type))

    with tf.variable_scope(var_scope_name):
      if self.wn_enabled:
        V_initializer = \
          tf.random_normal_initializer(mean=0, stddev=math.sqrt(dropout * 1.0 / in_dim))
        self.V = tf.get_variable(
            'V',
            shape=[in_dim, out_dim],
            initializer=V_initializer,
            trainable=True)
        self.V_norm = tf.norm(self.V.initialized_value(), axis=0)
        self.g = tf.get_variable('g', initializer=self.V_norm, trainable=True)
      else:
        self.V = tf.get_variable(
            'W',
            shape=[in_dim, out_dim],
            initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
            trainable=True)
      if self.bias_enabled:
        self.b = tf.get_variable(
            'b',
            shape=[out_dim],
            initializer=tf.zeros_initializer(),
            trainable=True)
      else:
        self.b = None

  def call(self, x):
    """Projects x with its linear transformation.

    Args:
      x: A float32 tensor with shape [batch_size, length, in_dim]

    Returns:
      float32 tensor with shape [batch_size, length, out_dim].
    """
    batch_size = tf.shape(x)[0]

    x = tf.reshape(x, [-1, self.in_dim])
    output = tf.matmul(x, self.V)
    output = tf.reshape(output, [batch_size, -1, self.out_dim])

    if self.wn_enabled:
      # x*(v*(g/2-norm(v)))
      scaler = tf.div(self.g, tf.norm(self.V, axis=0))
      output = tf.reshape(scaler, [1, self.out_dim]) * output

    if self.b is not None:
      output = output + tf.reshape(self.b, [1, self.out_dim])

    if self.apply_batch_norm:
      output = tf.expand_dims(output, axis=1)
      output = tf.layers.batch_normalization(
          name=self.var_scope_name + "_batch_norm",
          inputs=output,
          #gamma_regularizer=regularizer,
          training=self.mode == 'train',
          axis=-1,
          momentum=0.99,
          epsilon=1e-4,
      )
      output = tf.squeeze(output, axis=1)

    if self.apply_layer_norm:
      output = tf.expand_dims(output, axis=1)
      output = tf.contrib.layers.layer_norm(
          inputs=output,
          begin_norm_axis=1,
          begin_params_axis=-1,
          scope=self.var_scope_name + "_layer_norm",
      )
      output = tf.squeeze(output, axis=1)

    return output
