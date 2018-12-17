# This code is heavily based on the code from MLPerf
# https://github.com/mlperf/reference/tree/master/translation/tensorflow/transformer

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Define defaults for parameters
class Transformer_BatchNorm(tf.layers.Layer):
  """Transformer batch norn: supports [BTC] (default) and [BCT] formats. """

  def __init__(self, training, params={}):
        #data_format='channels_last', momentum = 0.9, epsilon = 0.0001):
    super(Transformer_BatchNorm, self).__init__()
    self.training = training
    self.data_format=params.get('data_format','channels_last')
    self.momentum = params.get('momentum',0.95)
    self.epsilon  = params.get('epsilon',0.0001)
    print("Batch norm, training=", training, params)


  def call(self, x):
    # x1 = tf.transpose(x, [0, 2, 1])  # B C T
    # x1 = tf.expand_dims(x1, -1)
    # y1 = tf.layers.batch_normalization(
    #   # name="{}/bn".format(name),
    #   center=True,
    #   scale=True, #False, #,
    #   inputs=x1,
    #   training= self.training,
    #   axis= 1,
    #   momentum= self.momentum,
    #   epsilon = self.epsilon,
    # )
    # y1 = tf.squeeze(y1, [3])
    # y2 = tf.transpose(y1, [0, 2, 1])

    #print("bn_input", x.shape)
    x1 = tf.expand_dims(x, axis=2)
    if (self.data_format=='channels_last'):
      axis=-1
    else:
      axis=1
    y1 = tf.layers.batch_normalization(
      center=True,
      scale=True,
      inputs=x1,
      training= self.training,
      axis= axis,
      momentum= self.momentum,
      epsilon = self.epsilon,
    )
    y2 = tf.squeeze(y1, axis=[2])
    return y2

class LayerNormalization(tf.layers.Layer):
  """Layer normalization for BTC format: supports L2(default) and L1 modes"""

  def __init__(self, hidden_size, params={}):
    #layer_norm_type="layernorm_L2", epsilon=1e-6):
    super(LayerNormalization, self).__init__()
    self.hidden_size = hidden_size
    self.norm_type = params.get("type", "layernorm_L2")
    self.epsilon = params.get("epsilon", 1e-6)
    print ("Layer norm, mode=", self.norm_type, params)

  def build(self, _):
    self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                 initializer=tf.ones_initializer(dtype=self.dtype),
                                 dtype=self.dtype)
    self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                initializer=tf.zeros_initializer(dtype=self.dtype),
                                dtype=self.dtype)
    self.built = True


  # def call(self, x, epsilon=1e-6):
  #   dtype = x.dtype
  #   x = tf.cast(x=x, dtype=tf.float32)
  #   mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
  #   variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
  #   norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
  #   result = norm_x * self.scale + self.bias
  #   return tf.cast(x=result, dtype=dtype)

  def call(self, x): # epsilon=1e-6):
    dtype = x.dtype
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    x = x - mean
    if self.norm_type=="layernorm_L2":
      # variance = tf.reduce_mean(tf.square(x), axis=[-1], keepdims=True)
      # norm_x = x * tf.rsqrt(variance + self.epsilon)
      if dtype==tf.float16:
        x = tf.cast(x, dtype=tf.float32)

      variance = tf.reduce_mean(tf.square(x), axis=[-1], keepdims=True)
      norm_x = x * tf.rsqrt(variance + self.epsilon)

      if dtype == tf.float16:
        norm_x= tf.saturate_cast(norm_x, dtype)

    elif self.norm_type=="layernorm_L1":
      variance = tf.reduce_mean(tf.abs(x), axis=[-1], keepdims=True)
      norm_x = tf.div(x , variance + self.epsilon)
    else:
      print("WARNING: Layer norm: type ", self.norm_type, "not supported")
      norm_x = x
    y = norm_x * self.scale + self.bias
    return y

class PrePostProcessingWrapper(object):
  """Wrapper around layer, that applies pre-processing and post-processing."""

  def __init__(self, layer, params, training):
    self.layer = layer
    self.postprocess_dropout = params["layer_postprocess_dropout"]
    self.training = training

    # Create normalization layer
    self.norm_params = params.get("norm_params", {"type": "layernorm_L2"})
    if self.norm_params["type"]=="batch_norm":
      self.norm = Transformer_BatchNorm(training=training,
                                        params=self.norm_params)
    elif self.norm_params["type"]=="layernorm_L2" or self.norm_params["type"]=="layernorm_L1":
      self.norm = LayerNormalization(hidden_size=params["hidden_size"],
                                     params=self.norm_params)
    else:
      print("WARNING: PrePostProcessingWrapper: unkonwn norm type=", self.norm_type)
      self.norm = LayerNormalization(hidden_size=params["hidden_size"],
                                     params=self.norm_params)


  def __call__(self, x, *args, **kwargs):
    # Preprocessing: normalization
    x_norm = self.norm(x)
    y = self.layer(x_norm, *args, **kwargs)
    # Postprocessing: dropout and residual connection
    if self.training:
      y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
    return x + y
