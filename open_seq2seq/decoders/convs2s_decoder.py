from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
import math
from .decoder import Decoder

from open_seq2seq.parts.transformer import beam_search

from open_seq2seq.parts.transformer import embedding_layer
from open_seq2seq.parts.transformer.utils import get_padding

from open_seq2seq.parts.convs2s import ffn_wn_layer, conv_wn_layer, attention_wn_layer
from open_seq2seq.parts.convs2s.utils import gated_linear_units

# Default value used if max_input_length is not given
MAX_INPUT_LENGTH = 128


class ConvS2SDecoder(Decoder):

  @staticmethod
  def get_required_params():
    """Static method with description of required parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **have to** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return dict(
        Decoder.get_required_params(), **{
            'batch_size': int,
            'tgt_emb_size': int,
            'tgt_vocab_size': int,
            'shared_embed': bool,
            'embedding_dropout_keep_prob': float,
            'conv_nchannels_kwidth': list,
            'hidden_dropout_keep_prob': float,
            'out_dropout_keep_prob': float,
            'beam_size': int,
            'alpha': float,
            'extra_decode_length': int,
            'EOS_ID': int,
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
    return dict(
        Decoder.get_optional_params(),
        **{
            'pad_embeddings_2_eight': bool,
            # set the default to False later.
            "pos_embed": bool,
            # if not provided, tgt_emb_size is used as the default value
            'out_emb_size': int,
            'max_input_length': int,
            'GO_SYMBOL': int,
            'PAD_SYMBOL': int,
            'END_SYMBOL': int,
            'conv_activation': None,
            'normalization_type': str,
            'scaling_factor': float,
            'init_var': None,
        })

  def _cast_types(self, input_dict):
    return input_dict

  def __init__(self, params, model, name="convs2s_decoder", mode='train'):
    super(ConvS2SDecoder, self).__init__(params, model, name, mode)
    self.embedding_softmax_layer = None
    self.position_embedding_layer = None
    self.layers = []
    self._tgt_vocab_size = self.params['tgt_vocab_size']
    self._tgt_emb_size = self.params['tgt_emb_size']
    self._mode = mode
    self._pad_sym = self.params.get('PAD_SYMBOL', 0)
    self._pad2eight = params.get('pad_embeddings_2_eight', False)
    self.scaling_factor = self.params.get("scaling_factor", math.sqrt(0.5))
    self.normalization_type = self.params.get("normalization_type", "weight_norm")
    self.conv_activation = self.params.get("conv_activation", gated_linear_units)
    self.max_input_length = self.params.get("max_input_length", MAX_INPUT_LENGTH)
    self.init_var = self.params.get('init_var', None)
    self.regularizer = self.params.get('regularizer', None)

  def _decode(self, input_dict):
    targets = input_dict['target_tensors'][0] \
              if 'target_tensors' in input_dict else None

    encoder_outputs = input_dict['encoder_output']['outputs']
    encoder_outputs_b = input_dict['encoder_output'].get(
        'outputs_b', encoder_outputs)

    inputs_attention_bias = input_dict['encoder_output'].get(
        'inputs_attention_bias_cs2s', None)

    with tf.name_scope("decode"):
      # prepare decoder layers
      if len(self.layers) == 0:
        knum_list = list(zip(*self.params.get("conv_nchannels_kwidth")))[0]
        kwidth_list = list(zip(*self.params.get("conv_nchannels_kwidth")))[1]

        # preparing embedding layers
        with tf.variable_scope("embedding"):
          if 'embedding_softmax_layer' in input_dict['encoder_output'] \
                  and self.params['shared_embed']:
            self.embedding_softmax_layer = \
              input_dict['encoder_output']['embedding_softmax_layer']
          else:
            self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
                vocab_size=self._tgt_vocab_size,
                hidden_size=self._tgt_emb_size,
                pad_vocab_to_eight=self._pad2eight,
                init_var=0.1,
                embed_scale=False,
                pad_sym=self._pad_sym,
                mask_paddings=True)

        if self.params.get("pos_embed", True):
          with tf.variable_scope("pos_embedding"):
            if 'position_embedding_layer' in input_dict['encoder_output'] \
                    and self.params['shared_embed']:
              self.position_embedding_layer = \
                input_dict['encoder_output']['position_embedding_layer']
            else:
              self.position_embedding_layer = embedding_layer.EmbeddingSharedWeights(
                  vocab_size=self.max_input_length,
                  hidden_size=self._tgt_emb_size,
                  pad_vocab_to_eight=self._pad2eight,
                  init_var=0.1,
                  embed_scale=False,
                  pad_sym=self._pad_sym,
                  mask_paddings=True)
        else:
          self.position_embedding_layer = None

        # linear projection before cnn layers
        self.layers.append(
            ffn_wn_layer.FeedFowardNetworkNormalized(
                self._tgt_emb_size,
                knum_list[0],
                dropout=self.params["embedding_dropout_keep_prob"],
                var_scope_name="linear_mapping_before_cnn_layers",
                mode=self.mode,
                normalization_type=self.normalization_type,
                regularizer=self.regularizer,
                init_var=self.init_var)
          )

        for i in range(len(knum_list)):
          in_dim = knum_list[i] if i == 0 else knum_list[i - 1]
          out_dim = knum_list[i]

          # linear projection is needed for residual connections if
          # input and output of a cnn layer do not match
          if in_dim != out_dim:
            linear_proj = ffn_wn_layer.FeedFowardNetworkNormalized(
                in_dim,
                out_dim,
                var_scope_name="linear_mapping_cnn_" + str(i + 1),
                dropout=1.0,
                mode=self.mode,
                normalization_type=self.normalization_type,
                regularizer = self.regularizer,
                init_var = self.init_var,
            )
          else:
            linear_proj = None

          conv_layer = conv_wn_layer.Conv1DNetworkNormalized(
              in_dim,
              out_dim,
              kernel_width=kwidth_list[i],
              mode=self.mode,
              layer_id=i + 1,
              hidden_dropout=self.params["hidden_dropout_keep_prob"],
              conv_padding="VALID",
              decode_padding=True,
              activation=self.conv_activation,
              normalization_type=self.normalization_type,
              regularizer=self.regularizer,
              init_var=self.init_var
          )

          att_layer = attention_wn_layer.AttentionLayerNormalized(
              out_dim,
              embed_size=self._tgt_emb_size,
              layer_id=i + 1,
              add_res=True,
              mode=self.mode,
              normalization_type=self.normalization_type,
              scaling_factor=self.scaling_factor,
              regularizer=self.regularizer,
              init_var=self.init_var
          )

          self.layers.append([linear_proj, conv_layer, att_layer])

        # linear projection after cnn layers
        self.layers.append(
            ffn_wn_layer.FeedFowardNetworkNormalized(
                knum_list[-1],
                self.params.get("out_emb_size", self._tgt_emb_size),
                dropout=1.0,
                var_scope_name="linear_mapping_after_cnn_layers",
                mode=self.mode,
                normalization_type=self.normalization_type,
                regularizer=self.regularizer,
                init_var=self.init_var))

        if not self.params['shared_embed']:
          self.layers.append(
              ffn_wn_layer.FeedFowardNetworkNormalized(
                  self.params.get("out_emb_size", self._tgt_emb_size),
                  self._tgt_vocab_size,
                  dropout=self.params["out_dropout_keep_prob"],
                  var_scope_name="linear_mapping_to_vocabspace",
                  mode=self.mode,
                  normalization_type=self.normalization_type,
                  regularizer=self.regularizer,
                  init_var=self.init_var))
        else:
          # if embedding is shared,
          # the shared embedding is used as the final linear projection to vocab space
          self.layers.append(None)

      if targets is None:
        return self.predict(encoder_outputs, encoder_outputs_b,
                            inputs_attention_bias)
      else:
        logits = self.decode_pass(targets, encoder_outputs, encoder_outputs_b,
                                  inputs_attention_bias)
      return {
          "logits": logits,
          "outputs": [tf.argmax(logits, axis=-1)],
          "final_state": None,
          "final_sequence_lengths": None
      }

  def decode_pass(self, targets, encoder_outputs, encoder_outputs_b,
                  inputs_attention_bias):
    """Generate logits for each value in the target sequence.

    Args:
      targets: target values for the output sequence.
        int tensor with shape [batch_size, target_length]
      encoder_outputs: continuous representation of input sequence.
        float tensor with shape [batch_size, input_length, hidden_size]
        float tensor with shape [batch_size, input_length, hidden_size]
      encoder_outputs_b: continuous representation of input sequence
        which includes the source embeddings.
        float tensor with shape [batch_size, input_length, hidden_size]
      inputs_attention_bias: float tensor with shape [batch_size, 1, input_length]

    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
    """

    # Prepare inputs to decoder layers by applying embedding
    # and adding positional encoding.
    decoder_inputs = self.embedding_softmax_layer(targets)

    if self.position_embedding_layer is not None:
      with tf.name_scope("add_pos_encoding"):
        pos_input = tf.range(
            0,
            tf.shape(decoder_inputs)[1],
            delta=1,
            dtype=tf.int32,
            name='range')
        pos_encoding = self.position_embedding_layer(pos_input)
        decoder_inputs = decoder_inputs + tf.cast(
            x=pos_encoding, dtype=decoder_inputs.dtype)

    if self.mode == "train":
      decoder_inputs = tf.nn.dropout(decoder_inputs,
                                     self.params["embedding_dropout_keep_prob"])

    # mask the paddings in the target
    inputs_padding = get_padding(
        targets, padding_value=self._pad_sym, dtype=decoder_inputs.dtype)
    decoder_inputs *= tf.expand_dims(1.0 - inputs_padding, 2)

    # do decode
    logits = self._call(
        decoder_inputs=decoder_inputs,
        encoder_outputs_a=encoder_outputs,
        encoder_outputs_b=encoder_outputs_b,
        input_attention_bias=inputs_attention_bias)

    return logits

  def _call(self, decoder_inputs, encoder_outputs_a, encoder_outputs_b,
            input_attention_bias):
    # run input into the decoder layers and returns the logits
    target_embed = decoder_inputs
    with tf.variable_scope("linear_layer_before_cnn_layers"):
      outputs = self.layers[0](decoder_inputs)

    for i in range(1, len(self.layers) - 2):
      linear_proj, conv_layer, att_layer = self.layers[i]

      with tf.variable_scope("layer_%d" % i):
        if linear_proj is not None:
          res_inputs = linear_proj(outputs)
        else:
          res_inputs = outputs

        with tf.variable_scope("conv_layer"):
          outputs = conv_layer(outputs)

        with tf.variable_scope("attention_layer"):
          outputs = att_layer(outputs, target_embed, encoder_outputs_a,
                              encoder_outputs_b, input_attention_bias)
        outputs = (outputs + res_inputs) * self.scaling_factor


    with tf.variable_scope("linear_layer_after_cnn_layers"):
      outputs = self.layers[-2](outputs)

    if self.mode == "train":
      outputs = tf.nn.dropout(outputs, self.params["out_dropout_keep_prob"])

    with tf.variable_scope("pre_softmax_projection"):
      if self.layers[-1] is None:
        logits = self.embedding_softmax_layer.linear(outputs)
      else:
        logits = self.layers[-1](outputs)

    return tf.cast(logits, dtype=tf.float32)

  def predict(self, encoder_outputs, encoder_outputs_b, inputs_attention_bias):
    """Return predicted sequence."""
    batch_size = tf.shape(encoder_outputs)[0]
    input_length = tf.shape(encoder_outputs)[1]

    max_decode_length = input_length + self.params["extra_decode_length"]

    symbols_to_logits_fn = self._get_symbols_to_logits_fn()

    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    initial_ids = tf.zeros(
        [batch_size], dtype=tf.int32) + self.params["GO_SYMBOL"]

    cache = {}
    # Add encoder outputs and attention bias to the cache.
    cache["encoder_outputs"] = encoder_outputs
    cache["encoder_outputs_b"] = encoder_outputs_b
    if inputs_attention_bias is not None:
      cache["inputs_attention_bias"] = inputs_attention_bias

    # Use beam search to find the top beam_size sequences and scores.
    decoded_ids, scores = beam_search.sequence_beam_search(
        symbols_to_logits_fn=symbols_to_logits_fn,
        initial_ids=initial_ids,
        initial_cache=cache,
        vocab_size=self.params["tgt_vocab_size"],
        beam_size=self.params["beam_size"],
        alpha=self.params["alpha"],
        max_decode_length=max_decode_length,
        eos_id=self.params["EOS_ID"])

    # Get the top sequence for each batch element
    top_decoded_ids = decoded_ids[:, 0, :]
    top_scores = scores[:, 0]

    # this isn't particularly efficient
    logits = self.decode_pass(top_decoded_ids, encoder_outputs,
                              encoder_outputs_b, inputs_attention_bias)

    return {
        "logits": logits,
        "outputs": [top_decoded_ids],
        "final_state": None,
        "final_sequence_lengths": None
    }

  def _get_symbols_to_logits_fn(self):
    """Returns a decoding function that calculates logits of the next tokens."""

    def symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next potential IDs.

      Args:
        ids: Current decoded sequences.
          int tensor with shape [batch_size * beam_size, i - 1]
        i: Loop index
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """

      # pass the decoded ids from the beginneing up to the current into the decoder
      # not efficient
      decoder_outputs = self.decode_pass(ids, cache.get("encoder_outputs"),
                                         cache.get("encoder_outputs_b"),
                                         cache.get("inputs_attention_bias"))

      logits = decoder_outputs[:, i, :]
      return logits, cache

    return symbols_to_logits_fn
