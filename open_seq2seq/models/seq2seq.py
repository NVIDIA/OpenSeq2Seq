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
  def __init__(self,
               params,
               data_layer,
               encoder,
               decoder,
               loss,
               global_step=None,
               force_var_reuse=False,
               mode=None,
               gpu_ids=None,
               hvd=None):
    """
    Constructor
    :param params: Python dictionary - parameters describing seq2seq model
    :param data_layer: Instance of DataLayer
    :param encoder: Instance of Encoder
    :param decoder: Instance of Decoder
    :param loss: Instance of Loss
    :param global_step: TF variable - global step
    :param force_var_reuse: Boolean - if true, all vars will be re-used
    :param mode: string, currently "train" or "infer"
    :param gpu_ids: a list of gpu ids, None, or "horovod" string
                    for distributed training using Horovod
    """
    # this has to happen before super call since this quantities are used in
    # _build_forward_pass_graph function which is called in the super init
    encoder.set_model(self)
    decoder.set_model(self)
    loss.set_model(self)

    self._encoder = encoder
    self._decoder = decoder
    self._loss_computator = loss

    super(Seq2Seq, self).__init__(params=params,
                                  data_layer=data_layer,
                                  global_step=global_step,
                                  force_var_reuse=force_var_reuse,
                                  mode=mode,
                                  gpu_ids=gpu_ids,
                                  hvd=hvd)

  def _build_forward_pass_graph(self,
                                input_tensors,
                                gpu_id=0):
    """
    Builds forward pass
    :param input_tensors: List of Tensors, currently assumes the following:
    [source_sequence, src_length, target_sequence, tgt_length]
    :param gpu_id: gpu_id where this pass is being built
    :return: loss or nothing
    """
    if self.mode == "infer":
      source_sequence, src_length = input_tensors
      target_sequence, tgt_length = None, None
    else:
      source_sequence, src_length, target_sequence, tgt_length = input_tensors

    with tf.variable_scope("ForwardPass"):
      encoder_input = {
        "src_inputs": source_sequence,
        "src_lengths": src_length,
      }
      encoder_output = self.encoder.encode(input_dict=encoder_input)

      # TODO: target length else part needs some comment
      decoder_input = {
        "encoder_output": encoder_output,
        "tgt_inputs": target_sequence,
        "tgt_lengths": tgt_length if self.mode == "train"
                                  else tf.cast(1.2 * tf.cast(src_length,tf.float32),
                                               tf.int32),
      }
      decoder_output = self.decoder.decode(input_dict=decoder_input)
      # TODO: better name?
      decoder_samples = decoder_output.get("decoder_samples", None)

      with tf.variable_scope("Loss"):
        if self.mode == "train" or self.mode == "eval":
          decoder_logits = decoder_output["decoder_output"]
          loss_input_dict = {
            "logits": decoder_logits,
            "target_sequence": target_sequence,
            "tgt_lengths": tgt_length,
          }
          loss = self.loss_computator.compute_loss(loss_input_dict)
        else:
          loss = None
          deco_print("Inference Mode. Loss part of graph isn't built.")
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
