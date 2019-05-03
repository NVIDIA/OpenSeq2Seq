# Copyright (c) 2019 NVIDIA Corporation
import tensorflow as tf
from tensorflow.python.ops import math_ops

from open_seq2seq.parts.centaur import AttentionBlock
from open_seq2seq.parts.centaur import ConvBlock
from open_seq2seq.parts.centaur import Prenet
from open_seq2seq.parts.transformer import utils
from open_seq2seq.parts.transformer.common import LayerNormalization
from .decoder import Decoder


class CentaurDecoder(Decoder):
  """
  Centaur decoder that consists of attention blocks
  followed by convolutional layers.
  """

  @staticmethod
  def get_required_params():
    return dict(Decoder.get_required_params(), **{
        "prenet_layers": int,
        "prenet_hidden_size": int,
        "hidden_size": int,
        "conv_layers": list,
        "mag_conv_layers": None,
        "attention_dropout": float,
        "layer_postprocess_dropout": float
    })

  @staticmethod
  def get_optional_params():
    return dict(Decoder.get_optional_params(), **{
        "prenet_activation_fn": None,
        "prenet_dropout": float,
        "prenet_use_inference_dropout": bool,
        "cnn_dropout_prob": float,
        "bn_momentum": float,
        "bn_epsilon": float,
        "reduction_factor": int,
        "attention_layers": int,
        "self_attention_conv_params": dict,
        "attention_heads": int,
        "attention_cnn_dropout_prob": float,
        "window_size": int,
        "back_step_size": int,
        "force_layers": list
    })

  def __init__(self, params, model, name="centaur_decoder", mode="train"):
    """
    Centaur decoder constructor.

    See parent class for arguments description.

    Config parameters:

    * **prenet_layers** (int) --- number of fully-connected layers to use.
    * **prenet_hidden_size** (int) --- number of units in each pre-net layer.
    * **hidden_size** (int) --- dimensionality of hidden embeddings.
    * **conv_layers** (list) --- list with the description of convolutional
      layers. For example::
        "conv_layers": [
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "VALID", "is_causal": True
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "VALID", "is_causal": True
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "VALID", "is_causal": True
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "VALID", "is_causal": True
          }
        ]
    * **mag_conv_layers** (list) --- list with the description of convolutional
      layers to reconstruct magnitude.
    * **attention_dropout** (float) --- dropout rate for attention layers.
    * **layer_postprocess_dropout** (float) --- dropout rate for
      transformer block sublayers.
    * **prenet_activation_fn** (callable) --- activation function to use for the
      prenet lyaers. Defaults to relu.
    * **prenet_dropout** (float) --- dropout rate for the pre-net. Defaults to 0.5.
    * **prenet_use_inference_dropout** (bool) --- whether to use dropout during the inference.
      Defaults to False.
    * **cnn_dropout_prob** (float) --- dropout probabilty for cnn layers.
      Defaults to 0.5.
    * **bn_momentum** (float) --- momentum for batch norm. Defaults to 0.95.
    * **bn_epsilon** (float) --- epsilon for batch norm. Defaults to 1e-8.
    * **reduction_factor** (int) --- number of frames to predict in a time.
      Defaults to 1.
    * **attention_layers** (int) --- number of attention blocks. Defaults to 4.
    * **self_attention_conv_params** (dict) --- description of convolutional
      layer inside attention blocks. Defaults to None.
    * **attention_heads** (int) --- number of attention heads. Defaults to 1.
    * **attention_cnn_dropout_prob** (float) --- dropout rate for convolutional
      layers inside attention blocks. Defaults to 0.5.
    * **window_size** (int) --- size of attention window for forcing
      monotonic attention during the inference. Defaults to None.
    * **back_step_size** (int) --- number of steps attention is allowed to
      go back during the inference. Defaults to 0.
    * **force_layers** (list) --- indices of layers where forcing of
      monotonic attention should be enabled. Defaults to all layers.
    """

    super(CentaurDecoder, self).__init__(params, model, name, mode)

    data_layer_params = model.get_data_layer().params
    n_feats = data_layer_params["num_audio_features"]
    use_mag = "both" in data_layer_params["output_type"]

    self.training = mode == "train"
    self.prenet = None
    self.linear_projection = None
    self.attentions = []
    self.output_normalization = None
    self.conv_layers = []
    self.mag_conv_layers = []
    self.stop_token_projection_layer = None
    self.mel_projection_layer = None
    self.mag_projection_layer = None

    self.n_mel = n_feats["mel"] if use_mag else n_feats
    self.n_mag = n_feats["magnitude"] if use_mag else None
    self.reduction_factor = params.get("reduction_factor", 1)

  def _build_layers(self):
    regularizer = self._params.get("regularizer", None)
    inference_dropout = self._params.get("prenet_use_inference_dropout", False)

    self.prenet = Prenet(
        n_layers=self._params["prenet_layers"],
        hidden_size=self._params["prenet_hidden_size"],
        activation_fn=self._params.get("prenet_activation_fn", tf.nn.relu),
        dropout=self._params.get("prenet_dropout", 0.5),
        regularizer=regularizer,
        training=self.training or inference_dropout,
        dtype=self._params["dtype"]
    )

    cnn_dropout_prob = self._params.get("cnn_dropout_prob", 0.5)
    bn_momentum = self._params.get("bn_momentum", 0.95)
    bn_epsilon = self._params.get("bn_epsilon", -1e8)

    self.linear_projection = tf.layers.Dense(
        name="linear_projection",
        units=self._params["hidden_size"],
        use_bias=False,
        kernel_regularizer=regularizer,
        dtype=self._params["dtype"]
    )

    n_layers = self._params.get("attention_layers", 4)
    n_heads = self._params.get("attention_heads", 1)
    conv_params = self._params.get("self_attention_conv_params", None)
    force_layers = self._params.get("force_layers", range(n_layers))

    for index in range(n_layers):
      window_size = None

      if index in force_layers:
        window_size = self._params.get("window_size", None)

      attention = AttentionBlock(
          name="attention_block_%d" % index,
          hidden_size=self._params["hidden_size"],
          attention_dropout=self._params["attention_dropout"],
          layer_postprocess_dropout=self._params["layer_postprocess_dropout"],
          regularizer=regularizer,
          training=self.training,
          cnn_dropout_prob=self._params.get("attention_cnn_dropout_prob", 0.5),
          conv_params=conv_params,
          n_heads=n_heads,
          window_size=window_size,
          back_step_size=self._params.get("back_step_size", None)
      )
      self.attentions.append(attention)

    self.output_normalization = LayerNormalization(self._params["hidden_size"])

    for index, params in enumerate(self._params["conv_layers"]):
      if params["num_channels"] == -1:
        params["num_channels"] = self.n_mel * self.reduction_factor

      layer = ConvBlock.create(
          index=index,
          conv_params=params,
          regularizer=regularizer,
          bn_momentum=bn_momentum,
          bn_epsilon=bn_epsilon,
          cnn_dropout_prob=cnn_dropout_prob,
          training=self.training
      )
      self.conv_layers.append(layer)

    for index, params in enumerate(self._params["mag_conv_layers"]):
      if params["num_channels"] == -1:
        params["num_channels"] = self.n_mag * self.reduction_factor

      layer = ConvBlock.create(
          index=index,
          conv_params=params,
          regularizer=regularizer,
          bn_momentum=bn_momentum,
          bn_epsilon=bn_epsilon,
          cnn_dropout_prob=cnn_dropout_prob,
          training=self.training
      )
      self.mag_conv_layers.append(layer)

    self.stop_token_projection_layer = tf.layers.Dense(
        name="stop_token_projection",
        units=1 * self.reduction_factor,
        use_bias=True,
        kernel_regularizer=regularizer
    )

    self.mel_projection_layer = tf.layers.Dense(
        name="mel_projection",
        units=self.n_mel * self.reduction_factor,
        use_bias=True,
        kernel_regularizer=regularizer
    )

    self.mag_projection_layer = tf.layers.Dense(
        name="mag_projection",
        units=self.n_mag * self.reduction_factor,
        use_bias=True,
        kernel_regularizer=regularizer
    )

  def _decode(self, input_dict):
    self._build_layers()

    if "target_tensors" in input_dict:
      targets = input_dict["target_tensors"][0]
    else:
      targets = None

    encoder_outputs = input_dict["encoder_output"]["outputs"]
    attention_bias = input_dict["encoder_output"]["inputs_attention_bias"]
    spec_length = None

    if self.mode == "train" or self.mode == "eval":
      spec_length = None

      if "target_tensors" in input_dict:
        spec_length = input_dict["target_tensors"][2]

    if self.training:
      return self._train(targets, encoder_outputs, attention_bias, spec_length)

    return self._infer(encoder_outputs, attention_bias, spec_length)

  def _decode_pass(self,
                   decoder_inputs,
                   encoder_outputs,
                   enc_dec_attention_bias,
                   sequence_lengths=None,
                   alignment_positions=None):
    y = self.prenet(decoder_inputs)
    y = self.linear_projection(y)

    with tf.variable_scope("decoder_pos_encoding"):
      pos_encoding = self._positional_encoding(y, self.params["dtype"])
      y += pos_encoding

    with tf.variable_scope("encoder_pos_encoding"):
      pos_encoding = self._positional_encoding(encoder_outputs, self.params["dtype"])
      encoder_outputs += pos_encoding

    for i, attention in enumerate(self.attentions):
      positions = None

      if alignment_positions is not None:
        positions = alignment_positions[i, :, :, :]

      y = attention(y, encoder_outputs, enc_dec_attention_bias, positions=positions)

    y = self.output_normalization(y)

    with tf.variable_scope("conv_layers"):
      for layer in self.conv_layers:
        y = layer(y)

    stop_token_logits = self.stop_token_projection_layer(y)
    mel_spec = self.mel_projection_layer(y)

    with tf.variable_scope("mag_conv_layers"):
      for layer in self.mag_conv_layers:
        y = layer(y)

    mag_spec = self.mag_projection_layer(y)

    if sequence_lengths is None:
      batch_size = tf.shape(y)[0]
      sequence_lengths = tf.zeros([batch_size])

    return {
        "spec": mel_spec,
        "post_net_spec": mel_spec,
        "alignments": None,
        "stop_token_logits": stop_token_logits,
        "lengths": sequence_lengths,
        "mag_spec": mag_spec
    }

  def _train(self, targets, encoder_outputs, enc_dec_attention_bias, sequence_lengths):
    # Shift targets to the right, and remove the last element
    with tf.name_scope("shift_targets"):
      n_features = self.n_mel + self.n_mag
      targets = targets[:, :, :n_features]
      targets = self._shrink(targets, n_features, self.reduction_factor)
      decoder_inputs = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

    outputs = self._decode_pass(
        decoder_inputs=decoder_inputs,
        encoder_outputs=encoder_outputs,
        enc_dec_attention_bias=enc_dec_attention_bias,
        sequence_lengths=sequence_lengths
    )

    with tf.variable_scope("alignments"):
      weights = []

      for index in range(len(self.attentions)):
        op = "ForwardPass/centaur_decoder/attention_block_%d/attention/attention/attention_weights" % index
        weights_operation = tf.get_default_graph().get_operation_by_name(op)
        weight = weights_operation.values()[0]
        weights.append(weight)

      outputs["alignments"] = [tf.stack(weights)]

    return self._convert_outputs(
        outputs,
        self.reduction_factor,
        self._model.params["batch_size_per_gpu"]
    )

  def _infer(self, encoder_outputs, enc_dec_attention_bias, sequence_lengths):
    if sequence_lengths is None:
      maximum_iterations = self._model.get_data_layer()._params.get("duration_max", 1000)
    else:
      maximum_iterations = tf.reduce_max(sequence_lengths)

    maximum_iterations //= self.reduction_factor

    state, state_shape_invariants = self._inference_initial_state(
        encoder_outputs,
        enc_dec_attention_bias
    )

    state = tf.while_loop(
        cond=self._inference_cond,
        body=self._inference_step,
        loop_vars=[state],
        shape_invariants=state_shape_invariants,
        back_prop=False,
        maximum_iterations=maximum_iterations,
        parallel_iterations=1
    )

    return self._convert_outputs(
        state["outputs"],
        self.reduction_factor,
        self._model.params["batch_size_per_gpu"]
    )

  def _inference_initial_state(self, encoder_outputs, encoder_decoder_attention_bias):
    """Create initial state for inference."""

    with tf.variable_scope("inference_initial_state"):
      batch_size = tf.shape(encoder_outputs)[0]
      n_layers = self._params.get("attention_layers", 1)
      n_heads = self._params.get("attention_heads", 1)
      n_features = self.n_mel + self.n_mag

      state = {
          "iteration": tf.constant(0),
          "inputs": tf.zeros([batch_size, 1, n_features * self.reduction_factor]),
          "finished": tf.cast(tf.zeros([batch_size]), tf.bool),
          "alignment_positions": tf.zeros([n_layers, batch_size, n_heads, 1], dtype=tf.int32),
          "outputs": {
              "spec": tf.zeros([batch_size, 0, self.n_mel * self.reduction_factor]),
              "post_net_spec": tf.zeros([batch_size, 0, self.n_mel * self.reduction_factor]),
              "alignments": [
                  tf.zeros([0, 0, 0, 0, 0])
              ],
              "stop_token_logits": tf.zeros([batch_size, 0, 1 * self.reduction_factor]),
              "lengths": tf.zeros([batch_size], dtype=tf.int32),
              "mag_spec": tf.zeros([batch_size, 0, self.n_mag * self.reduction_factor])
          },
          "encoder_outputs": encoder_outputs,
          "encoder_decoder_attention_bias": encoder_decoder_attention_bias
      }

      state_shape_invariants = {
          "iteration": tf.TensorShape([]),
          "inputs": tf.TensorShape([None, None, n_features * self.reduction_factor]),
          "finished": tf.TensorShape([None]),
          "alignment_positions": tf.TensorShape([n_layers, None, n_heads, None]),
          "outputs": {
              "spec": tf.TensorShape([None, None, self.n_mel * self.reduction_factor]),
              "post_net_spec": tf.TensorShape([None, None, self.n_mel * self.reduction_factor]),
              "alignments": [
                  tf.TensorShape([None, None, None, None, None]),
              ],
              "stop_token_logits": tf.TensorShape([None, None, 1 * self.reduction_factor]),
              "lengths": tf.TensorShape([None]),
              "mag_spec": tf.TensorShape([None, None, None])
          },
          "encoder_outputs": encoder_outputs.shape,
          "encoder_decoder_attention_bias": encoder_decoder_attention_bias.shape
      }

      return state, state_shape_invariants

  def _inference_cond(self, state):
    """Check if it's time to stop inference."""

    with tf.variable_scope("inference_cond"):
      all_finished = math_ops.reduce_all(state["finished"])
      return tf.logical_not(all_finished)

  def _inference_step(self, state):
    """Make one inference step."""

    decoder_inputs = state["inputs"]
    encoder_outputs = state["encoder_outputs"]
    enc_dec_attention_bias = state["encoder_decoder_attention_bias"]
    alignment_positions = state["alignment_positions"]

    outputs = self._decode_pass(
        decoder_inputs=decoder_inputs,
        encoder_outputs=encoder_outputs,
        enc_dec_attention_bias=enc_dec_attention_bias,
        alignment_positions=alignment_positions
    )

    with tf.variable_scope("inference_step"):
      next_inputs_mel = outputs["post_net_spec"][:, -1:, :]
      next_inputs_mel = self._expand(next_inputs_mel, self.reduction_factor)
      next_inputs_mag = outputs["mag_spec"][:, -1:, :]
      next_inputs_mag = self._expand(next_inputs_mag, self.reduction_factor)
      next_inputs = tf.concat([next_inputs_mel, next_inputs_mag], axis=-1)

      n_features = self.n_mel + self.n_mag
      next_inputs = self._shrink(next_inputs, n_features, self.reduction_factor)

      # Set zero if sequence is finished
      next_inputs = tf.where(
          state["finished"],
          tf.zeros_like(next_inputs),
          next_inputs
      )
      next_inputs = tf.concat([decoder_inputs, next_inputs], 1)

      # Update lengths
      lengths = state["outputs"]["lengths"]
      lengths = tf.where(
          state["finished"],
          lengths,
          lengths + 1 * self.reduction_factor
      )
      outputs["lengths"] = lengths

      # Update spec, post_net_spec and mag_spec
      for key in ["spec", "post_net_spec", "mag_spec"]:
        output = outputs[key][:, -1:, :]
        output = tf.where(state["finished"], tf.zeros_like(output), output)
        outputs[key] = tf.concat([state["outputs"][key], output], 1)

      # Update stop token logits
      stop_token_logits = outputs["stop_token_logits"][:, -1:, :]
      stop_token_logits = tf.where(
          state["finished"],
          tf.zeros_like(stop_token_logits) + 1e9,
          stop_token_logits
      )
      stop_prediction = tf.sigmoid(stop_token_logits)
      stop_prediction = tf.reduce_max(stop_prediction, axis=-1)

      # Uncomment next line if you want to use stop token predictions
      finished = tf.reshape(tf.cast(tf.round(stop_prediction), tf.bool), [-1])
      finished = tf.reshape(finished, [-1])

      stop_token_logits = tf.concat(
          [state["outputs"]["stop_token_logits"], stop_token_logits],
          axis=1
      )
      outputs["stop_token_logits"] = stop_token_logits

      with tf.variable_scope("alignments"):
        forward = "ForwardPass" if self.mode == "infer" else "ForwardPass_1"
        weights = []

        for index in range(len(self.attentions)):
          op = forward + "/centaur_decoder/while/attention_block_%d/attention/attention/attention_weights" % index
          weights_operation = tf.get_default_graph().get_operation_by_name(op)
          weight = weights_operation.values()[0]
          weights.append(weight)

        weights = tf.stack(weights)
        outputs["alignments"] = [weights]

      alignment_positions = tf.argmax(
          weights,
          axis=-1,
          output_type=tf.int32
      )[:, :, :, -1:]
      state["alignment_positions"] = tf.concat(
          [state["alignment_positions"], alignment_positions],
          axis=-1
      )

      state["iteration"] = state["iteration"] + 1
      state["inputs"] = next_inputs
      state["finished"] = finished
      state["outputs"] = outputs

    return state

  @staticmethod
  def _shrink(values, last_dim, reduction_factor):
    """Shrink the given input by reduction_factor."""

    shape = tf.shape(values)
    new_shape = [
        shape[0],
        shape[1] // reduction_factor,
        last_dim * reduction_factor
    ]
    values = tf.reshape(values, new_shape)
    return values

  @staticmethod
  def _expand(values, reduction_factor):
    """Expand the given input by reduction_factor."""

    shape = tf.shape(values)
    new_shape = [
        shape[0],
        shape[1] * reduction_factor,
        shape[2] // reduction_factor
    ]
    values = tf.reshape(values, new_shape)
    return values

  @staticmethod
  def _positional_encoding(x, dtype):
    """Add positional encoding to the given input."""

    length = tf.shape(x)[1]
    features_count = tf.shape(x)[2]
    features_count += features_count % 2
    pos_encoding = utils.get_position_encoding(length, features_count)
    position_encoding = tf.cast(pos_encoding, dtype)
    position_encoding = position_encoding[:, :features_count]
    return position_encoding

  @staticmethod
  def _convert_outputs(outputs, reduction_factor, batch_size):
    """Convert output of the decoder to appropriate format."""

    with tf.variable_scope("output_converter"):
      for key in ["spec", "post_net_spec", "stop_token_logits", "mag_spec"]:
        outputs[key] = CentaurDecoder._expand(outputs[key], reduction_factor)

      alignments = []
      for sample in range(batch_size):
        alignments.append([outputs["alignments"][0][:, sample, :, :, :]])

      return {
          "outputs": [
              outputs["spec"],
              outputs["post_net_spec"],
              alignments,
              tf.sigmoid(outputs["stop_token_logits"]),
              outputs["lengths"],
              outputs["mag_spec"]
          ],
          "stop_token_prediction": outputs["stop_token_logits"]
      }
