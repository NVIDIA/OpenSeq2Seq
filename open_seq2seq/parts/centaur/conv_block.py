import tensorflow as tf

from .batch_norm import BatchNorm1D


class ConvBlock:
  def __init__(self, name, conv, norm, activation_fn, dropout, training, is_residual, is_causal):
    self.name = name
    self.conv = conv
    self.norm = norm
    self.activation_fn = activation_fn
    self.dropout = dropout
    self.training = training
    self.is_residual = is_residual
    self.is_casual = is_causal

  def __call__(self, x):
    with tf.variable_scope(self.name):
      if self.is_casual:
        # Add padding from the left side to avoid looking to the future
        pad_size = self.conv.kernel_size[0] - 1
        y = tf.pad(x, [[0, 0], [pad_size, 0], [0, 0]])
      else:
        y = x

      y = self.conv(y)

      if self.norm is not None:
        y = self.norm(y, training=self.training)

      if self.activation_fn is not None:
        y = self.activation_fn(y)

      if self.dropout is not None:
        y = self.dropout(y, training=self.training)

      return x + y if self.is_residual else y

  @staticmethod
  def create(index,
             conv_params,
             regularizer,
             bn_momentum,
             bn_epsilon,
             cnn_dropout_prob,
             training,
             is_residual=True,
             is_causal=False):
    activation_fn = conv_params.get("activation_fn", tf.nn.relu)

    conv = tf.layers.Conv1D(
      name="conv_%d" % index,
      filters=conv_params["num_channels"],
      kernel_size=conv_params["kernel_size"],
      strides=conv_params["stride"],
      padding=conv_params["padding"],
      kernel_regularizer=regularizer
    )

    norm = BatchNorm1D(
      name="bn_%d" % index,
      gamma_regularizer=regularizer,
      momentum=bn_momentum,
      epsilon=bn_epsilon
    )

    dropout = tf.layers.Dropout(
      name="dropout_%d" % index,
      rate=cnn_dropout_prob
    )

    if "is_causal" in conv_params:
      is_causal = conv_params["is_causal"]

    if "is_residual" in conv_params:
      is_residual = conv_params["is_residual"]

    return ConvBlock("layer_%d" % index, conv, norm, activation_fn, dropout, training, is_residual, is_causal)
