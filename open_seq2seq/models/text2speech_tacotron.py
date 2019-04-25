# Copyright (c) 2019 NVIDIA Corporation

from .text2speech import Text2Speech


class Text2SpeechTacotron(Text2Speech):
  """
  Text-to-speech data layer for Tacotron.
  """

  def get_alignments(self, attention_mask):
    specs = [attention_mask]
    titles = ["alignments"]
    return specs, titles
