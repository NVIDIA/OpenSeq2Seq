# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import string
import os
import pandas as pd

if __name__ == '__main__':
  synthetic_data_root = "/data/speech/librispeech-syn/"
  synthetic_data_sample = synthetic_data_root + "{{}}/sample_step0_{}_syn.wav"

  in_char = "\"'’“”àâèéêü"
  out_char = "'''''aaeeeu"
  punctuation = string.punctuation.replace("'", "")
  table = str.maketrans(in_char, out_char, punctuation)

  def _normalize_transcript(text):
    """Parses the transcript to remove punctation, lowercase all characters, and
       all non-ascii characters

    Args:
      text: the string to parse

    Returns:
      text: the normalized text
    """
    text = text.translate(table)
    text = text.lower()
    text = text.strip()
    return text

  names = ["wav_filename", "wav_filesize", "transcript"]

  generated_files = pd.read_csv(
      "generate.csv", encoding='utf-8', sep='\x7c',
      header=None, quoting=3, names=names)
  num_files = len(generated_files)
  for i, row in enumerate(generated_files.itertuples()):
    generated_files.iat[i, 0] = synthetic_data_sample.format(i)
    line = _normalize_transcript(generated_files.iat[i, 2])
    generated_files.iat[i, 1] = -1
    generated_files.iat[i, 2] = line
    if i % int(num_files/10) == 0:
      print("Processed {} out of {}".format(i, num_files))
  generated_files.to_csv(
      os.path.join(synthetic_data_root, "synthetic_data.csv"), encoding='utf-8',
      sep=',', quoting=3, index=False)
