"""Implementation of a 1d convolutional layer with weight normalization.
Inspired from https://github.com/tobyyouup/conv_seq2seq"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import math
from open_seq2seq.parts.convs2s.utils import gated_linear_units
from open_seq2seq.parts.transformer.common import LayerNormalization


class Conv1DNetworkNormalized(tf.layers.Layer):
  """1D convolutional layer with weight normalization"""

  def __init__(self,
               in_dim,
               out_dim,
               kernel_width,
               mode,
               layer_id,
               hidden_dropout,
               conv_padding,
               decode_padding,
               activation=gated_linear_units,
               normalization_type="weight_norm",
               regularizer=None, # tf.contrib.layers.l2_regularizer(scale=1e-4)
               init_var=None,
               ):
    """initializes the 1D convolution layer.
    It uses weight normalization (Salimans & Kingma, 2016)  w = g * v/2-norm(v)

    Args:
      in_dim: int last dimension of the inputs
      out_dim: int new dimension for the output
      kernel_width: int width of kernel
      mode: str the current mode
      layer_id: int the id of current convolution layer
      hidden_dropout: float the keep-dropout value used on the input.
                      Give 1.0 if no dropout.
                      It is used to initialize the weights of convolution.
      conv_padding: str the type of padding done for convolution
      decode_padding: bool specifies if this convolution layer is in decoder or not
                          in decoder padding is done explicitly before convolution
      activation: the activation function applies after the convolution
      normalization_type: str specifies the normalization used for the layer.
                    "weight_norm" for weight normalization or
                    "batch_norm" for batch normalization or
                    "layer_norm" for layer normalization
      regularizer: the regularizer for the batch normalization

    """

    super(Conv1DNetworkNormalized, self).__init__()
    self.mode = mode
    self.conv_padding = conv_padding
    self.decode_padding = decode_padding
    self.hidden_dropout = hidden_dropout
    self.kernel_width = kernel_width
    self.layer_id = layer_id
    self.act_func = activation
    self.regularizer = regularizer

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
      self.bias_enabled = False
      self.wn_enabled = False
      self.apply_layer_norm = True
    elif normalization_type is None:
      self.apply_batch_norm = False
      self.bias_enabled = True
      self.wn_enabled = False
      self.apply_layer_norm = False
    else:
      raise ValueError("Wrong normalization type: {}".format(normalization_type))

    if activation == gated_linear_units:
      conv_out_size = 2 * out_dim
    else:
      conv_out_size = out_dim

    with tf.variable_scope("conv_layer_" + str(layer_id)):
      if init_var is None:
        V_std = math.sqrt(4.0 * hidden_dropout / (kernel_width * in_dim))
      else:
        V_std = init_var
      if self.wn_enabled:
        self.V = tf.get_variable(
            'V',
            shape=[kernel_width, in_dim, conv_out_size],
            initializer=tf.random_normal_initializer(mean=0, stddev=V_std),
            trainable=True)
        self.V_norm = tf.norm(self.V.initialized_value(), axis=[0, 1])
        self.g = tf.get_variable('g', initializer=self.V_norm, trainable=True)
        self.W = tf.reshape(self.g, [1, 1, conv_out_size]) * tf.nn.l2_normalize(
            self.V, [0, 1])
      else:
        self.W = tf.get_variable(
            'W',
            shape=[kernel_width, in_dim, conv_out_size],
            initializer=tf.random_normal_initializer(mean=0, stddev=V_std),
            trainable=True,
            regularizer=self.regularizer)

      if self.bias_enabled:
        self.b = tf.get_variable(
            'b',
            shape=[conv_out_size],
            initializer=tf.zeros_initializer(),
            trainable=True)
      else:
        self.b = None

      if self.apply_layer_norm:
        self.layer_norm = LayerNormalization(out_dim)
      else:
        self.layer_norm = None


  def call(self, input):
    """Applies convolution with gated linear units on x.

    Args:
      x: A float32 tensor with shape [batch_size, length, in_dim]

    Returns:
      float32 tensor with shape [batch_size, length, out_dim].
    """
    output = input

    if self.mode == "train":
      output = tf.nn.dropout(output, self.hidden_dropout)

    if self.decode_padding:
      output = tf.pad(
          output, [[0, 0], [self.kernel_width - 1, self.kernel_width - 1], [0, 0]],
          "CONSTANT")

    output = tf.nn.conv1d(
        value=output, filters=self.W, stride=1, padding=self.conv_padding)

    if self.decode_padding and self.kernel_width > 1:
      output = output[:, 0:-self.kernel_width + 1, :]

    if self.apply_batch_norm:
      # trick to make batchnorm work for mixed precision training.
      bn_input = tf.expand_dims(output, axis=1)
      bn_output = tf.layers.batch_normalization(
          name="batch_norm_" + str(self.layer_id),
          inputs=bn_input,
          training=self.mode == 'train',
          axis=-1,
          momentum=0.95,
          epsilon=1e-4
      )
      output = tf.squeeze(bn_output, axis=1)

    if self.apply_layer_norm:
      output = self.layer_norm(output)
    if self.b is not None:
      output = tf.nn.bias_add(output, self.b)

    if self.act_func is not None:
      output = self.act_func(output)
    return output
