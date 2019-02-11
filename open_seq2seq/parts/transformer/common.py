# This code is heavily based on the code from MLPerf
# https://github.com/mlperf/reference/tree/master/translation/tensorflow/transformer

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Transformer_BatchNorm(tf.layers.Layer):
  """Transformer batch norn: supports [BTC](default) and [BCT] formats. """

  def __init__(self, training, params={}):
    super(Transformer_BatchNorm, self).__init__()
    self.training = training
    self.data_format=params.get('data_format','channels_last')
    self.momentum = params.get('momentum',0.95)
    self.epsilon  = params.get('epsilon',0.0001)
    self.center_scale = params.get('center_scale', True)
    self.regularizer = params.get('regularizer', None) if self.center_scale else None
    if self.regularizer != None:
      self.regularizer_params = params.get("regularizer_params", {'scale': 0.0})
      self.regularizer=self.regularizer(self.regularizer_params['scale']) \
        if self.regularizer_params['scale'] > 0.0 else None

    #print("Batch norm, training=", training, params)

  def call(self, x):
    x = tf.expand_dims(x, axis=2)
    axis = -1 if (self.data_format=='channels_last') else 1
    y = tf.layers.batch_normalization(inputs=x, axis=axis,
      momentum=self.momentum, epsilon=self.epsilon,
      center=self.center_scale, scale=self.center_scale,
      beta_regularizer=self.regularizer, gamma_regularizer=self.regularizer,
      training=self.training,
    )
    y = tf.squeeze(y, axis=[2])
    return y

class LayerNormalization(tf.layers.Layer):
  """Layer normalization for BTC format: supports L2(default) and L1 modes"""

  def __init__(self, hidden_size, params={}):
    super(LayerNormalization, self).__init__()
    self.hidden_size = hidden_size
    self.norm_type = params.get("type", "layernorm_L2")
    self.epsilon = params.get("epsilon", 1e-6)

  def build(self, _):
    self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                 initializer= tf.keras.initializers.Ones(),
                                 dtype=tf.float32)
    self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                 initializer=tf.keras.initializers.Zeros(),
                                 dtype=tf.float32)
    self.built = True

  def call(self, x):
    if self.norm_type=="layernorm_L2":
      epsilon = self.epsilon
      dtype = x.dtype
      x = tf.cast(x=x, dtype=tf.float32)
      mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
      variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
      norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
      result = norm_x * self.scale + self.bias
      return tf.cast(x=result, dtype=dtype)

    else:
      dtype = x.dtype
      if dtype==tf.float16:
        x = tf.cast(x, dtype=tf.float32)
      mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
      x = x - mean
      variance = tf.reduce_mean(tf.abs(x), axis=[-1], keepdims=True)
      norm_x = tf.div(x , variance + self.epsilon)
      y = norm_x * self.scale + self.bias
      if dtype == tf.float16:
        y = tf.saturate_cast(y, dtype)
      return y

class PrePostProcessingWrapper(object):
  """Wrapper around layer, that applies pre-processing and post-processing."""

  def __init__(self, layer, params, training):
    self.layer = layer
    self.postprocess_dropout = params["layer_postprocess_dropout"]
    self.training = training
    self.norm_params = params.get("norm_params", {"type": "layernorm_L2"})
    # Create normalization layer
    if self.norm_params["type"]=="batch_norm":
      self.norm = Transformer_BatchNorm(training=training,
                                        params=self.norm_params)
    else:
      self.norm = LayerNormalization(hidden_size=params["hidden_size"],
                                     params=self.norm_params)

  def __call__(self, x, *args, **kwargs):
    # Preprocessing: normalization
    y = self.norm(x)
    y = self.layer(y, *args, **kwargs)
    # Postprocessing: dropout and residual connection
    if self.training:
      y = tf.nn.dropout(y, keep_prob=1 - self.postprocess_dropout)
    return x + y
