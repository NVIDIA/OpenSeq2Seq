# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
from open_seq2seq.utils.utils import deco_print
from open_seq2seq.models.model import Model


class Seq2Seq(Model):
  """
  Standard sequence-to-sequence class with one encoder and one decoder.
  "encoder-decoder-loss" models should inherit from this class.
  """

  @staticmethod
  def get_required_params():
    return dict(Model.get_required_params(), **{
      'encoder': None,  # could be any user defined class
      'decoder': None,  # could be any user defined class
    })

  @staticmethod
  def get_optional_params():
    return dict(Model.get_optional_params(), **{
      'encoder_params': dict,
      'decoder_params': dict,
      'loss': None,  # could be any user defined class
      'loss_params': dict,
    })

  def __init__(self, params, mode="train", hvd=None):
    """Sequence-to-sequence model constructor.
    Note that TensorFlow graph should not be created here. All graph creation
    logic is happening inside
    :meth:`self._build_forward_pass_graph() <_build_forward_pass_graph>` method.

    Args:
      params (dict): parameters describing the model.
          All supported parameters are listed in :meth:`get_required_params`,
          :meth:`get_optional_params` functions.
      mode (string, optional): "train", "eval" or "infer".
          If mode is "train" all parts of the graph will be built
          (model, loss, optimizer).
          If mode is "eval", only model and loss will be built.
          If mode is "infer", only model will be built.
      hvd (optional): if Horovod is used, this should be
          ``horovod.tensorflow`` module.
          If Horovod is not used, it should be None.

    Config parameters:

    * **encoder** (any class derived from
      :class:`Encoder <encoders.encoder.Encoder>`) --- encoder class to use.
    * **encoder_params** (dict) --- dictionary with encoder configuration. For
      complete list of possible parameters see the corresponding class docs.
    * **decoder** (any class derived from
      :class:`Decoder <decoders.decoder.Decoder>`) --- decoder class to use.
    * **decoder_params** (dict) --- dictionary with decoder configuration. For
      complete list of possible parameters see the corresponding class docs.
    * **loss** (any class derived from
      :class:`Loss <losses.loss.Loss>`) --- loss class to use.
    * **loss_params** (dict) --- dictionary with loss configuration. For
      complete list of possible parameters see the corresponding class docs.
    """
    super(Seq2Seq, self).__init__(params=params, mode=mode, hvd=hvd)
    if 'encoder_params' not in self.params:
      self.params['encoder_params'] = {}
    if 'decoder_params' not in self.params:
      self.params['decoder_params'] = {}
    if 'loss_params' not in self.params:
      self.params['loss_params'] = {}

    self._encoder = self._create_encoder()
    self._decoder = self._create_decoder()
    if self.mode == 'train' or self.mode == 'eval':
      self._loss_computator = self._create_loss()
    else:
      self._loss_computator = None

  def _create_encoder(self):
    """This function should return encoder class.
    Overwrite this function if additional parameters need to be specified for
    encoder, besides provided in the config.

    Returns:
      instance of a class derived from :class:`encoders.encoder.Encoder`.
    """
    params = self.params['encoder_params']
    return self.params['encoder'](params=params, mode=self.mode, model=self)

  def _create_decoder(self):
    """This function should return decoder class.
    Overwrite this function if additional parameters need to be specified for
    decoder, besides provided in the config.

    Returns:
      instance of a class derived from :class:`decoders.decoder.Decoder`.
    """
    params = self.params['decoder_params']
    params['tgt_vocab_size'] = self.data_layer.params['tgt_vocab_size']
    return self.params['decoder'](params=params, mode=self.mode, model=self)

  def _create_loss(self):
    """This function should return loss class.
    Overwrite this function if additional parameters need to be specified for
    loss, besides provided in the config.

    Returns:
      instance of a class derived from :class:`losses.loss.Loss`.
    """
    return self.params['loss'](params=self.params['loss_params'], model=self)

  def _build_forward_pass_graph(self, input_tensors, gpu_id=0):
    """TensorFlow graph for sequence-to-sequence model is created here.
    This function connects encoder, decoder and loss together. As an input for
    encoder it will specify source sequence and source length (as returned from
    the data layer). As an input for decoder it will specify target sequence
    and target length as well as all output returned from encoder. For loss it
    will also specify target sequence and length and all output returned from
    decoder. Note that loss will only be built for mode == "train" or "eval".

    See :meth:`models.model.Model._build_forward_pass_graph` for description of
    arguments and return values.
    """
    if self.mode == "infer":
      src_sequence, src_length = input_tensors
      tgt_sequence, tgt_length = None, None
    else:
      src_sequence, src_length, tgt_sequence, tgt_length = input_tensors

    with tf.variable_scope("ForwardPass"):
      encoder_input = {
        "src_sequence": src_sequence,
        "src_length": src_length,
      }
      encoder_output = self.encoder.encode(input_dict=encoder_input)

      tgt_length_eval = tf.cast(1.2 * tf.cast(src_length, tf.float32), tf.int32)
      decoder_input = {
        "encoder_output": encoder_output,
        "tgt_sequence": tgt_sequence,
        # when the mode is not "train", replacing correct tgt_length with
        # somewhat increased src_length
        "tgt_length": tgt_length if self.mode == "train" else tgt_length_eval
      }
      decoder_output = self.decoder.decode(input_dict=decoder_input)
      decoder_samples = decoder_output.get("samples", None)

      if self.mode == "train" or self.mode == "eval":
        with tf.variable_scope("Loss"):
          loss_input_dict = {
            "decoder_output": decoder_output,
            "tgt_sequence": tgt_sequence,
            "tgt_length": tgt_length,
          }
          loss = self.loss_computator.compute_loss(loss_input_dict)
      else:
        deco_print("Inference Mode. Loss part of graph isn't built.")
        loss = None
      return loss, decoder_samples

  @property
  def encoder(self):
    """Model encoder."""
    return self._encoder

  @property
  def decoder(self):
    """Model decoder."""
    return self._decoder

  @property
  def loss_computator(self):
    """Model loss computator."""
    return self._loss_computator
