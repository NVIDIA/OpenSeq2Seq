# Copyright (c) 2017 NVIDIA Corporation
import io


def load_pre_existing_vocabulary(path, min_idx=0, read_chars=False):
  """Loads pre-existing vocabulary into memory.
  The vocabulary file should contain a token on each line with optional
  token count on the same line that will be ignored. Example::

    a 1234
    b 4321
    c 32342
    d
    e
    word 234

  Args:
    path (str): path to vocabulary.
    min_idx (int, optional): minimum id to assign for a token.
    read_chars (bool, optional): whether to read only the
        first symbol of the line.

  Returns:
     dict: vocabulary dictionary mapping tokens (chars/words) to int ids.
  """
  idx = min_idx
  vocab_dict = {}
  with io.open(path, newline='', encoding='utf-8') as f:
    for line in f:
      # ignoring empty lines
      if not line or line == '\n':
        continue
      if read_chars:
        token = line[0]
      else:
        token = line.rstrip().split('\t')[0]
      vocab_dict[token] = idx
      idx += 1
  return vocab_dict
