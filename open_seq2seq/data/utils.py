# Copyright (c) 2017 NVIDIA Corporation
import random
from . import data_layer
import re

def pretty_print_array(row, vocab, ignore_special=False, delim=' '):
  n = len(vocab)
  if ignore_special:
    f_row = []
    for i in range(0, len(row)):
      char_id = row[i]
      if char_id==data_layer.DataLayer.EOS_ID:
        break
      if char_id!=data_layer.DataLayer.PAD_ID and char_id!=data_layer.DataLayer.S_ID:
        f_row += [char_id]
    return delim.join(map(lambda x: vocab[x], [r for r in f_row if r > 0 and r < n]))
  else:
    return delim.join(map(lambda x: vocab[x], [r for r in row if r > 0 and r < n]))


def weighted_choice(choices):
  total_weights = sum(w for c, w in choices.items())
  if total_weights <= 0:
    return data_layer.DataLayer.END_OF_CHOICE
  r = random.uniform(0, total_weights)
  mx = 0
  for i, w in choices.items():
    if mx + w >= r:
      return i
    mx += w
  raise AssertionError("weighted choice got to the wrong place")


def transform_for_bleu(row, vocab, ignore_special=False, delim=' ', bpe_used=False):
  n = len(vocab)
  if ignore_special:
    f_row = []
    for i in range(0, len(row)):
      char_id = row[i]
      if char_id==data_layer.DataLayer.EOS_ID:
        break
      if char_id!=data_layer.DataLayer.PAD_ID and char_id!=data_layer.DataLayer.S_ID:
        f_row += [char_id]
    sentence = [vocab[r] for r in f_row if r > 0 and r < n]
  else:
    sentence = [vocab[r] for r in row if r > 0 and r < n]

  if bpe_used:
    sentence = delim.join(sentence)
    sentence = re.sub("@@ ", "", sentence)
    sentence = sentence.split(delim)

  return  sentence
