"""Implementation of a 1d convolutional layer with weight normalization.
Inspired from https://github.com/tobyyouup/conv_seq2seq"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import math


class Conv1DNetworkNormalized(tf.layers.Layer):
  """1D convolutional layer with weight normalization"""

  def __init__(self, in_dim, out_dim, kernel_width, mode, layer_id,
               hidden_dropout, conv_padding, decode_padding):
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
    """

    super(Conv1DNetworkNormalized, self).__init__()
    self.mode = mode
    self.conv_padding = conv_padding
    self.decode_padding = decode_padding
    self.hidden_dropout = hidden_dropout
    self.kernel_width = kernel_width

    with tf.variable_scope("conv_layer_" + str(layer_id)):
      V_std = math.sqrt(4.0 * hidden_dropout / (kernel_width * in_dim))
      self.V = tf.get_variable(
          'V',
          shape=[kernel_width, in_dim, 2 * out_dim],
          initializer=tf.random_normal_initializer(mean=0, stddev=V_std),
          trainable=True)
      self.V_norm = tf.norm(self.V.initialized_value(), axis=[0, 1])
      self.g = tf.get_variable('g', initializer=self.V_norm, trainable=True)
      self.b = tf.get_variable(
          'b',
          shape=[2 * out_dim],
          initializer=tf.zeros_initializer(),
          trainable=True)

      self.W = tf.reshape(self.g, [1, 1, 2 * out_dim]) * tf.nn.l2_normalize(
          self.V, [0, 1])

  def call(self, input):
    """Applies convolution with gated linear units on x.

    Args:
      x: A float32 tensor with shape [batch_size, length, in_dim]

    Returns:
      float32 tensor with shape [batch_size, length, out_dim].
    """
    x = input
    if self.mode == "train":
      x = tf.nn.dropout(x, self.hidden_dropout)

    if self.decode_padding:
      x = tf.pad(
          x, [[0, 0], [self.kernel_width - 1, self.kernel_width - 1], [0, 0]],
          "CONSTANT")

    output = tf.nn.bias_add(
        tf.nn.conv1d(
            value=x, filters=self.W, stride=1, padding=self.conv_padding),
        self.b)

    if self.decode_padding and self.kernel_width > 1:
      output = output[:, 0:-self.kernel_width + 1, :]

    output = self.gated_linear_units(output)

    return output

  def gated_linear_units(self, inputs):
    """Gated Linear Units (GLU) on x.

    Args:
      x: A float32 tensor with shape [batch_size, length, 2*out_dim]
    Returns:
      float32 tensor with shape [batch_size, length, out_dim].
    """
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) == 3
    input_pass = inputs[:, :, 0:int(input_shape[2] / 2)]
    input_gate = inputs[:, :, int(input_shape[2] / 2):]
    input_gate = tf.sigmoid(input_gate)
    return tf.multiply(input_pass, input_gate)
