# This code is heavily based on the code from MLPerf
# https://github.com/mlperf/reference/tree/master/translation/tensorflow/transformer

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
from six.moves import range

from open_seq2seq.parts.transformer import utils, attention_layer, \
                                           ffn_layer, beam_search
from open_seq2seq.parts.transformer.common import PrePostProcessingWrapper, \
                                                  LayerNormalization
from .decoder import Decoder


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
    return dict(Decoder.get_required_params(), **{
        'EOS_ID': int,
        'layer_postprocess_dropout': float,
        'num_hidden_layers': int,
        'hidden_size': int,
        'num_heads': int,
        'attention_dropout': float,
        'relu_dropout': float,
        'filter_size': int,
        'batch_size': int,
        'tgt_vocab_size': int,
        'beam_size': int,
        'alpha': float,
        'extra_decode_length': int,
    })

  @staticmethod
  def get_optional_params():
    """Static method with description of optional parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **can** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return dict(Decoder.get_optional_params(), **{
        'regularizer': None,  # any valid TensorFlow regularizer
        'regularizer_params': dict,
        'initializer': None,  # any valid TensorFlow initializer
        'initializer_params': dict,
        'GO_SYMBOL': int,
        'PAD_SYMBOL': int,
        'END_SYMBOL': int,
    })

  def _cast_types(self, input_dict):
    return input_dict

  def __init__(self, params, model,
               name="transformer_decoder", mode='train'):
    super(TransformerDecoder, self).__init__(params, model, name, mode)
    self.embedding_softmax_layer = None
    self.output_normalization = None
    self._mode = mode
    self.layers = []

  def _decode(self, input_dict):
    if 'target_tensors' in input_dict:
      targets = input_dict['target_tensors'][0]
    else:
      targets = None
    encoder_outputs = input_dict['encoder_output']['outputs']
    inputs_attention_bias = (
      input_dict['encoder_output']['inputs_attention_bias']
    )
    self.embedding_softmax_layer = (
      input_dict['encoder_output']['embedding_softmax_layer']
    )

    with tf.name_scope("decode"):
      # prepare decoder layers
      if len(self.layers) == 0:
        for _ in range(self.params["num_hidden_layers"]):
          self_attention_layer = attention_layer.SelfAttention(
              self.params["hidden_size"], self.params["num_heads"],
              self.params["attention_dropout"],
              self.mode == "train",
          )
          enc_dec_attention_layer = attention_layer.Attention(
              self.params["hidden_size"], self.params["num_heads"],
              self.params["attention_dropout"],
              self.mode == "train",
          )
          feed_forward_network = ffn_layer.FeedFowardNetwork(
              self.params["hidden_size"], self.params["filter_size"],
              self.params["relu_dropout"], self.mode == "train",
          )

          self.layers.append([
              PrePostProcessingWrapper(self_attention_layer, self.params,
                                       self.mode == "train"),
              PrePostProcessingWrapper(enc_dec_attention_layer, self.params,
                                       self.mode == "train"),
              PrePostProcessingWrapper(feed_forward_network, self.params,
                                       self.mode == "train")
          ])

        self.output_normalization = LayerNormalization(
            self.params["hidden_size"]
        )

      if targets is None:
        return self.predict(encoder_outputs, inputs_attention_bias)
      else:
        logits = self.decode_pass(targets, encoder_outputs,
                                  inputs_attention_bias)
        return {"logits": logits,
                "outputs": [tf.argmax(logits, axis=-1)],
                "final_state": None,
                "final_sequence_lengths": None}

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
          # TODO: Figure out why this is needed
          # decoder_self_attention_bias = tf.cast(x=decoder_self_attention_bias,
          #                                      dtype=decoder_inputs.dtype)
          decoder_inputs = self_attention_layer(
              decoder_inputs, decoder_self_attention_bias, cache=layer_cache,
          )
        with tf.variable_scope("encdec_attention"):
          decoder_inputs = enc_dec_attention_layer(
              decoder_inputs, encoder_outputs, attention_bias,
          )
        with tf.variable_scope("ffn"):
          decoder_inputs = feed_forward_network(decoder_inputs)

    return self.output_normalization(decoder_inputs)

  def decode_pass(self, targets, encoder_outputs, inputs_attention_bias):
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
    # Prepare inputs to decoder layers by shifting targets, adding positional
    # encoding and applying dropout.
    decoder_inputs = self.embedding_softmax_layer(targets)
    with tf.name_scope("shift_targets"):
      # Shift targets to the right, and remove the last element
      decoder_inputs = tf.pad(
          decoder_inputs, [[0, 0], [1, 0], [0, 0]],
      )[:, :-1, :]
    with tf.name_scope("add_pos_encoding"):
      length = tf.shape(decoder_inputs)[1]
      # decoder_inputs += utils.get_position_encoding(
      #    length, self.params["hidden_size"])
      decoder_inputs += tf.cast(
          utils.get_position_encoding(length, self.params["hidden_size"]),
          dtype=self.params['dtype'],
      )
    if self.mode == "train":
      decoder_inputs = tf.nn.dropout(
          decoder_inputs, 1 - self.params["layer_postprocess_dropout"],
      )

    # Run values
    decoder_self_attention_bias = utils.get_decoder_self_attention_bias(length)

    # do decode
    outputs = self._call(
        decoder_inputs=decoder_inputs,
        encoder_outputs=encoder_outputs,
        decoder_self_attention_bias=decoder_self_attention_bias,
        attention_bias=inputs_attention_bias,
    )

    logits = self.embedding_softmax_layer.linear(outputs)
    return logits

  def _get_symbols_to_logits_fn(self, max_decode_length):
    """Returns a decoding function that calculates logits of the next tokens."""

    timing_signal = utils.get_position_encoding(
        max_decode_length + 1, self.params["hidden_size"],
    )
    decoder_self_attention_bias = utils.get_decoder_self_attention_bias(
        max_decode_length,
    )

    def symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next potential IDs.

      Args:
        ids: Current decoded sequences.
          int tensor with shape [batch_size * beam_size, i + 1]
        i: Loop index
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      # Set decoder input to the last generated IDs
      decoder_input = ids[:, -1:]

      # Preprocess decoder input by getting embeddings and adding timing signal.
      decoder_input = self.embedding_softmax_layer(decoder_input)
      decoder_input += tf.cast(x=timing_signal[i:i + 1],
                               dtype=decoder_input.dtype)

      self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      decoder_outputs = self._call(
          decoder_input, cache.get("encoder_outputs"), self_attention_bias,
          cache.get("encoder_decoder_attention_bias"), cache,
      )
      logits = self.embedding_softmax_layer.linear(decoder_outputs)
      logits = tf.squeeze(logits, axis=[1])
      return tf.cast(logits, tf.float32), cache

    return symbols_to_logits_fn

  def predict(self, encoder_outputs, encoder_decoder_attention_bias):
    """Return predicted sequence."""
    batch_size = tf.shape(encoder_outputs)[0]
    input_length = tf.shape(encoder_outputs)[1]
    max_decode_length = input_length + self.params["extra_decode_length"]

    symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    # Create cache storing decoder attention values for each layer.
    cache = {
        "layer_%d" % layer: {
            "k": tf.zeros([batch_size, 0,
                           self.params["hidden_size"]],
                          dtype=encoder_outputs.dtype),
            "v": tf.zeros([batch_size, 0,
                           self.params["hidden_size"]],
                          dtype=encoder_outputs.dtype),
        } for layer in range(self.params["num_hidden_layers"])
    }

    # Add encoder output and attention bias to the cache.
    cache["encoder_outputs"] = encoder_outputs
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    # Use beam search to find the top beam_size sequences and scores.
    decoded_ids, scores = beam_search.sequence_beam_search(
        symbols_to_logits_fn=symbols_to_logits_fn,
        initial_ids=initial_ids,
        initial_cache=cache,
        vocab_size=self.params["tgt_vocab_size"],
        beam_size=self.params["beam_size"],
        alpha=self.params["alpha"],
        max_decode_length=max_decode_length,
        eos_id=self.params["EOS_ID"],
    )

    # Get the top sequence for each batch element
    top_decoded_ids = decoded_ids[:, 0, 1:]

    # this isn't particularly efficient
    logits = self.decode_pass(top_decoded_ids, encoder_outputs,
                              encoder_decoder_attention_bias)
    return {"logits": logits,
            "outputs": [top_decoded_ids],
            "final_state": None,
            "final_sequence_lengths": None}
