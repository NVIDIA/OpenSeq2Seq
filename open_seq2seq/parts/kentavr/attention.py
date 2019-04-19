import tensorflow as tf

from open_seq2seq.parts.kentavr import ConvBlock
from open_seq2seq.parts.transformer import attention_layer
from open_seq2seq.parts.transformer.common import PrePostProcessingWrapper
from open_seq2seq.parts.transformer.ffn_layer import FeedFowardNetwork


class AttentionBlock:
  def __init__(self,
               hidden_size,
               attention_dropout,
               layer_postprocess_dropout,
               training,
               cnn_dropout_prob,
               regularizer=None,
               conv_params=None,
               n_heads=1,
               window_size=None,
               back_step_size=None,
               name="attention_block"):
    self.name = name
    self.conv = None

    if conv_params:
      self.conv = ConvBlock.create(
        index=0,
        conv_params=conv_params,
        regularizer=regularizer,
        bn_momentum=0.95,
        bn_epsilon=1e-8,
        cnn_dropout_prob=cnn_dropout_prob,
        training=training
      )
      self.conv.name = "conv"

    attention = attention_layer.Attention(
      hidden_size=hidden_size,
      num_heads=n_heads,
      attention_dropout=attention_dropout,
      regularizer=regularizer,
      train=training,
      window_size=window_size,
      back_step_size=back_step_size,
    )

    feed_forward = FeedFowardNetwork(
      hidden_size=hidden_size,
      filter_size=4 * hidden_size,
      relu_dropout=0,
      regularizer=regularizer,
      train=training
    )

    wrapper_params = {
      "hidden_size": hidden_size,
      "layer_postprocess_dropout": layer_postprocess_dropout
    }

    self.attention = PrePostProcessingWrapper(
      layer=attention,
      params=wrapper_params,
      training=training
    )

    self.feed_forward = PrePostProcessingWrapper(
      layer=feed_forward,
      params=wrapper_params,
      training=training
    )

  def __call__(self, decoder_inputs, encoder_outputs, attention_bias, positions=None):
    with tf.variable_scope(self.name):
      y = decoder_inputs

      if self.conv:
        y = self.conv(y)

      with tf.variable_scope("attention"):
        y = self.attention(y, encoder_outputs, attention_bias, positions=positions)

      with tf.variable_scope("feed_forward"):
        y = self.feed_forward(y)

      return y
