import tensorflow as tf

from open_seq2seq.encoders import Encoder
from open_seq2seq.parts.centaur import ConvBlock
from open_seq2seq.parts.transformer import embedding_layer
from open_seq2seq.parts.transformer import utils


class CentaurEncoder(Encoder):
  """
  Centaur encoder that consists of convolutional layers.
  """

  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
        "src_vocab_size": int,
        "embedding_size": int,
        "output_size": int,
        "conv_layers": list
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
        "pad_embeddings_2_eight": bool,
        "regularizer": None,
        "bn_momentum": float,
        "bn_epsilon": float,
        "cnn_dropout_prob": float,
        "norm_type": str
    })

  def __init__(self, params, model, name="centaur_encoder", mode="train"):
    """
    Centaur encoder constructor.

    See parent class for arguments description.

    Config parameters:

    * **src_vocab_size** (int) --- number of symbols in alphabet.
    * **embedding_size** (int) --- dimensionality of character embedding.
    * **output_size** (int) --- dimensionality of output embedding.
    * **conv_layers** (list) --- list with the description of convolutional
      layers. For example::
        "conv_layers": [
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME"
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME"
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME"
          }
        ]
    * **bn_momentum** (float) --- momentum for batch norm. Defaults to 0.95.
    * **bn_epsilon** (float) --- epsilon for batch norm. Defaults to 1e-8.
    * **cnn_dropout_prob** (float) --- dropout probabilty for cnn layers.
      Defaults to 0.5.

    """

    super(CentaurEncoder, self).__init__(params, model, name=name, mode=mode)
    self.training = mode == "train"
    self.layers = []

  def _build_layers(self):
    regularizer = self._params.get("regularizer", None)

    embedding = embedding_layer.EmbeddingSharedWeights(
        vocab_size=self._params["src_vocab_size"],
        hidden_size=self._params["embedding_size"],
        pad_vocab_to_eight=self.params.get("pad_embeddings_2_eight", False),
        regularizer=regularizer
    )
    self.layers.append(embedding)

    cnn_dropout_prob = self._params.get("cnn_dropout_prob", 0.5)
    bn_momentum = self._params.get("bn_momentum", 0.95)
    bn_epsilon = self._params.get("bn_epsilon", -1e8)

    for index, params in enumerate(self._params["conv_layers"]):
      layer = ConvBlock.create(
          index=index,
          conv_params=params,
          regularizer=regularizer,
          bn_momentum=bn_momentum,
          bn_epsilon=bn_epsilon,
          cnn_dropout_prob=cnn_dropout_prob,
          training=self.training
      )

      self.layers.append(layer)

    linear_projection = tf.layers.Dense(
        name="linear_projection",
        units=self._params["output_size"],
        use_bias=False,
        kernel_regularizer=regularizer
    )
    self.layers.append(linear_projection)

  def _encode(self, input_dict):
    if not self.layers:
      self._build_layers()

    x = input_dict["source_tensors"][0]
    text_len = input_dict["source_tensors"][1]

    # Apply all layers
    y = x
    for layer in self.layers:
      y = layer(y)

    inputs_attention_bias = utils.get_padding_bias(x)

    return {
        "outputs": y,
        "inputs_attention_bias": inputs_attention_bias,
        "src_lengths": text_len
    }
