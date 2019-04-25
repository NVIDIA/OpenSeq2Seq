# Copyright (c) 2019 NVIDIA Corporation

from six.moves import range

from .text2speech import Text2Speech


class Text2SpeechCentaur(Text2Speech):
  """
  Text-to-speech data layer for Centaur.
  """

  def get_alignments(self, attention_mask):
    alignments_name = ["dec_enc_alignment"]

    specs = []
    titles = []

    for name, alignment in zip(alignments_name, attention_mask):
      for layer in range(len(alignment)):
        for head in range(alignment.shape[1]):
          specs.append(alignment[layer][head])
          titles.append("{}_layer_{}_head_{}".format(name, layer, head))

    return specs, titles
