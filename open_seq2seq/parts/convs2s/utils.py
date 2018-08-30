"""Implementation of a 1d convolutional layer with weight normalization.
Inspired from https://github.com/tobyyouup/conv_seq2seq"""

import tensorflow as tf


def gated_linear_units(inputs):
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
