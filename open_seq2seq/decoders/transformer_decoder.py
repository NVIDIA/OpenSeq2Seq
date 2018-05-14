# This code is heavily based on the code from MLPerf
# https://github.com/mlperf/reference/tree/master/translation/tensorflow/transformer

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
from .decoder import Decoder
from open_seq2seq.parts.transformer import utils, attention_layer, ffn_layer
from open_seq2seq.parts.transformer.common import PrePostProcessingWrapper, \
  LayerNormalization

class TransformerDecoder(Decoder):
  @staticmethod
  def get_required_params():
    """Static method with description of required parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **have to** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return {}

  @staticmethod
  def get_optional_params():
    """Static method with description of optional parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **can** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return {
      'regularizer': None,  # any valid TensorFlow regularizer
      'regularizer_params': dict,
      'initializer': None,  # any valid TensorFlow initializer
      'initializer_params': dict,
      'dtype': [tf.float32, tf.float16, 'mixed'],
      'layer_postprocess_dropout': float,
      'num_hidden_layers': int,
      'hidden_size': int,
      'num_heads': int,
      'attention_dropout': float,
      'relu_dropout': float,
      'filter_size': int,
      'batch_size': int,
      'tgt_vocab_size': int
    }

  def __init__(self, params, model,
               name="transformer_decoder", mode='train'):
    super(TransformerDecoder, self).__init__(params, model, name, mode)
    self.embedding_softmax_layer = None
    self.output_normalization = None
    self._mode = mode
    self.layers = []

  def _call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
           attention_bias, cache=None):
    for n, layer in enumerate(self.layers):
      self_attention_layer = layer[0]
      enc_dec_attention_layer = layer[1]
      feed_forward_network = layer[2]

      # Run inputs through the sublayers.
      layer_name = "layer_%d" % n
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          decoder_inputs = self_attention_layer(
              decoder_inputs, decoder_self_attention_bias, cache=layer_cache)
        with tf.variable_scope("encdec_attention"):
          decoder_inputs = enc_dec_attention_layer(
              decoder_inputs, encoder_outputs, attention_bias)
        with tf.variable_scope("ffn"):
          decoder_inputs = feed_forward_network(decoder_inputs)

    return self.output_normalization(decoder_inputs)


  def predict(self, encoder_outputs, inputs_attention_bias):
    # TODO: Implement
    pass

  def train_decode(self, targets, encoder_outputs, inputs_attention_bias):
    """Generate logits for each value in the target sequence.

    Args:
      targets: target values for the output sequence.
        int tensor with shape [batch_size, target_length]
      encoder_outputs: continuous representation of input sequence.
        float tensor with shape [batch_size, input_length, hidden_size]
      inputs_attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
    """
    with tf.name_scope("decode"):
      # Prepare inputs to decoder layers by shifting targets, adding positional
      # encoding and applying dropout.
      decoder_inputs = self.embedding_softmax_layer(targets)
      with tf.name_scope("shift_targets"):
        # Shift targets to the right, and remove the last element
        decoder_inputs = tf.pad(
            decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(decoder_inputs)[1]
        decoder_inputs += utils.get_position_encoding(
            length, self.params["hidden_size"])
      if self.mode == "train":
        decoder_inputs = tf.nn.dropout(
            decoder_inputs, 1 - self.params["layer_postprocess_dropout"])

      # Run values
      decoder_self_attention_bias = utils.get_decoder_self_attention_bias(
          length)

      # prepare decoder layers
      if len(self.layers) == 0:
        for _ in range(self.params["num_hidden_layers"]):
          self_attention_layer = attention_layer.SelfAttention(
            self.params["hidden_size"], self.params["num_heads"],
            self.params["attention_dropout"],
            self.mode == "train")
          enc_dec_attention_layer = attention_layer.Attention(
            self.params["hidden_size"], self.params["num_heads"],
            self.params["attention_dropout"],
            self.mode == "train")
          feed_forward_network = ffn_layer.FeedFowardNetwork(
            self.params["hidden_size"], self.params["filter_size"],
            self.params["relu_dropout"], self.mode == "train")

          self.layers.append([
            PrePostProcessingWrapper(self_attention_layer, self.params,
                                     self.mode == "train"),
            PrePostProcessingWrapper(enc_dec_attention_layer, self.params,
                                     self.mode == "train"),
            PrePostProcessingWrapper(feed_forward_network, self.params,
                                     self.mode == "train")])

        self.output_normalization = LayerNormalization(self.params["hidden_size"])

      # do decode
      outputs = self._call(decoder_inputs=decoder_inputs,
                           encoder_outputs=encoder_outputs,
                           decoder_self_attention_bias=decoder_self_attention_bias,
                           attention_bias=inputs_attention_bias)

      logits = self.embedding_softmax_layer.linear(outputs)
      return {"logits": logits,
              "samples": [tf.argmax(logits, axis=-1)],
              "final_state": None,
              "final_sequence_lengths": None}



  def _decode(self, input_dict):
    targets = input_dict['tgt_sequence']
    encoder_outputs = input_dict['encoder_output']['outputs']
    inputs_attention_bias = input_dict['encoder_output']['inputs_attention_bias']
    self.embedding_softmax_layer = input_dict['encoder_output']['embedding_softmax_layer']

    if targets is None:
      return self.predict(encoder_outputs, inputs_attention_bias)
    else:
      return self.train_decode(targets, encoder_outputs, inputs_attention_bias)

