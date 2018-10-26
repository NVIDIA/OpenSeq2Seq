# Copyright (c) 2018 NVIDIA Corporation
import numpy as np
from scipy.io.wavfile import write

from .encoder_decoder import EncoderDecoderModel

def save_audio(signal, logdir, step, sampling_rate, mode):
  signal = np.float32(signal)
  file_name = '{}/sample_step{}_{}.wav'.format(logdir, step, mode)
  if logdir[0] != '/':
    file_name = "./" + file_name
  write(file_name, sampling_rate, signal)

class Text2SpeechWavenet(EncoderDecoderModel):

  @staticmethod
  def get_required_params():
    return dict(
        EncoderDecoderModel.get_required_params(), **{}
    )

  def __init__(self, params, mode="train", hvd=None):
    super(Text2SpeechWavenet, self).__init__(params, mode=mode, hvd=hvd)

  def maybe_print_logs(self, input_values, output_values, training_step):
    save_audio(
        output_values[1][-1],
        self.params["logdir"],
        training_step,
        sampling_rate=22050,
        mode="train"
    )
    return {}

  def evaluate(self, input_values, output_values):
    return output_values[1][-1]

  def finalize_evaluation(self, results_per_batch, training_step=None):
    save_audio(
        results_per_batch[0],
        self.params["logdir"],
        training_step,
        sampling_rate=22050,
        mode="eval"
    )
    return {}

  def infer(self, input_values, output_values):
    return output_values[1][-1]

  def finalize_inference(self, results_per_batch, output_file):
    return {}
