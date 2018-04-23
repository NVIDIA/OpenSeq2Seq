# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from open_seq2seq.utils.utils import deco_print
from open_seq2seq.models.model import Model


class Seq2Seq(Model):
  """
  Standard Sequence-to-Sequence class with one encoder and one decoder.
  "encoder-decoder-loss" models should inherit from this
  """

  @staticmethod
  def get_required_params():
    """Static method with description of required parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **have to** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return dict(Model.get_required_params(), **{
      'encoder': None,  # could be any user defined class
      'decoder': None,  # could be any user defined class
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
    return dict(Model.get_optional_params(), **{
      'encoder_params': dict,
      'decoder_params': dict,
      'loss': None,  # could be any user defined class
      'loss_params': dict,
    })

  def __init__(self, params, mode="train", hvd=None):
    super(Seq2Seq, self).__init__(params=params, mode=mode, hvd=hvd)

    encoder_params = self.params.get('encoder_params', {})
    encoder_params['batch_size'] = self.params['batch_size_per_gpu']
    self._encoder = self.params['encoder'](
      params=encoder_params, mode=mode, model=self,
    )
    decoder_params = self.params.get('decoder_params', {})
    decoder_params['batch_size'] = self.params['batch_size_per_gpu']
    self._decoder = self.params['decoder'](
      params=decoder_params, mode=mode, model=self,
    )
    if self.mode == 'train' or self.mode == 'eval':
      loss_params = self.params.get('loss_params', {})
      loss_params['batch_size'] = self.params['batch_size_per_gpu']
      self._loss_computator = self.params['loss'](
        params=loss_params, model=self,
      )
    else:
      self._loss_computator = None

  def _build_forward_pass_graph(self, input_tensors, gpu_id=0):
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


      decoder_input = {
        "encoder_output": encoder_output,
        "tgt_sequence": tgt_sequence,
        # TODO: why????
        "tgt_length": tgt_length if self.mode == "train"
                                 else tf.cast(1.2 * tf.cast(src_length,tf.float32),
                                               tf.int32),
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
    return self._encoder

  @property
  def decoder(self):
    return self._decoder

  @property
  def loss_computator(self):
    return self._loss_computator
