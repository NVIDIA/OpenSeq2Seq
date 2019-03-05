# This code is heavily based on the code from MLPerf
# https://github.com/mlperf/reference/tree/master/translation/tensorflow
# /transformer
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
from six.moves import range

from open_seq2seq.encoders import Encoder
from open_seq2seq.parts.transformer import attention_layer, ffn_layer, utils, \
                                           embedding_layer
from open_seq2seq.parts.transformer.common import PrePostProcessingWrapper, \
                                    LayerNormalization, Transformer_BatchNorm


class TransformerEncoder(Encoder):
  """Transformer model encoder"""

  @staticmethod
  def get_required_params():
    """Static method with description of required parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **have to** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return dict(Encoder.get_required_params(), **{
        "encoder_layers": int,
        "hidden_size": int,
        "num_heads": int,
        "attention_dropout": float,
        "filter_size": int,
        "src_vocab_size": int,
        "relu_dropout": float,
        "layer_postprocess_dropout": float,
        "remove_padding": bool,
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
    return dict(Encoder.get_optional_params(), **{
        'regularizer': None,  # any valid TensorFlow regularizer
        'regularizer_params': dict,
        'initializer': None,  # any valid TensorFlow initializer
        'initializer_params': dict,
        'pad_embeddings_2_eight': bool,
        'norm_params': dict,
    })

  def __init__(self, params, model, name="transformer_encoder", mode='train' ):
    super(TransformerEncoder, self).__init__(
        params, model, name=name, mode=mode,
    )
    self.layers = []
    self.output_normalization = None
    self._mode = mode

    self.embedding_softmax_layer = None
    self.norm_params = self.params.get("norm_params", {"type": "layernorm_L2"})
    self.regularizer = self.params.get("regularizer", None)
    if self.regularizer != None:
      self.regularizer_params = params.get("regularizer_params", {'scale': 0.0})
      self.regularizer=self.regularizer(self.regularizer_params['scale']) \
        if self.regularizer_params['scale'] > 0.0 else None


  def _call(self, encoder_inputs, attention_bias, inputs_padding):
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.variable_scope("layer_%d" % n):
        with tf.variable_scope("self_attention"):
          encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
        with tf.variable_scope("ffn"):
          encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)

    return self.output_normalization(encoder_inputs)

  def _encode(self, input_dict):
    training = (self.mode == "train")

    if len(self.layers) == 0:
      # prepare encoder graph
      self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
          self.params["src_vocab_size"], self.params["hidden_size"],
          pad_vocab_to_eight=self.params.get('pad_embeddings_2_eight', False),
      )

      for _ in range(self.params['encoder_layers']):
        # Create sublayers for each layer.
        self_attention_layer = attention_layer.SelfAttention(
          hidden_size=self.params["hidden_size"],
          num_heads=self.params["num_heads"],
          attention_dropout=self.params["attention_dropout"],
          train=training,
          regularizer=self.regularizer
        )
        feed_forward_network = ffn_layer.FeedFowardNetwork(
          hidden_size=self.params["hidden_size"],
          filter_size=self.params["filter_size"],
          relu_dropout=self.params["relu_dropout"],
          train=training,
          regularizer=self.regularizer
        )

        self.layers.append([
            PrePostProcessingWrapper(self_attention_layer, self.params,
                                     training),
            PrePostProcessingWrapper(feed_forward_network, self.params,
                                     training)
        ])

      # final normalization layer.
      print("Encoder:", self.norm_params["type"], self.mode)
      if self.norm_params["type"] =="batch_norm":
        self.output_normalization = Transformer_BatchNorm(
          training=training,
          params=self.norm_params)
      else:
        self.output_normalization = LayerNormalization(
          hidden_size=self.params["hidden_size"], params=self.norm_params)

    # actual encoder part
    with tf.name_scope("encode"):
      inputs = input_dict['source_tensors'][0]
      # Prepare inputs to the layer stack by adding positional encodings and
      # applying dropout.
      embedded_inputs = self.embedding_softmax_layer(inputs)
      if self.params["remove_padding"]:
        inputs_padding = utils.get_padding(inputs)
        #inputs_padding = utils.get_padding(inputs,dtype=self._params["dtype"])
      else:
        inputs_padding = None
      inputs_attention_bias = utils.get_padding_bias(inputs)
      # inputs_attention_bias = utils.get_padding_bias(inputs, dtype=self._params["dtype"])

      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(embedded_inputs)[1]
        pos_encoding = utils.get_position_encoding(
            length, self.params["hidden_size"],
        )
        encoder_inputs = embedded_inputs + tf.cast(x=pos_encoding,
                                                   dtype=embedded_inputs.dtype)

      if self.mode == "train":
        encoder_inputs = tf.nn.dropout(encoder_inputs,
            keep_prob = 1.0 - self.params["layer_postprocess_dropout"],
        )

      encoded = self._call(encoder_inputs, inputs_attention_bias,
                           inputs_padding)
      return {'outputs': encoded,
              'inputs_attention_bias': inputs_attention_bias,
              'state': None,
              'src_lengths': input_dict['source_tensors'][1],
              'embedding_softmax_layer': self.embedding_softmax_layer,
              'encoder_input': inputs}
