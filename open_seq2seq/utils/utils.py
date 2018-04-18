# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import tensorflow as tf


def mask_nans(x):
  x_zeros = tf.zeros_like(x)
  x_mask = tf.is_finite(x)
  y = tf.where(x_mask, x, x_zeros)
  return y


def deco_print(line, offset=0, start="*** ", end='\n'):
  print(start + " " * offset + line, end=end)


def array_to_string(row, vocab, delim=' '):
  n = len(vocab)
  return delim.join(map(lambda x: vocab[x], [r for r in row if 0 <= r < n]))


def text_ids_to_string(row, vocab, S_ID, EOS_ID, PAD_ID,
                       ignore_special=False, delim=' '):
  """For _-to-text outputs this function takes a row with ids,
  target vocabulary and prints it as a human-readable string
  """
  n = len(vocab)
  if ignore_special:
    f_row = []
    for i in range(0, len(row)):
      char_id = row[i]
      if char_id == EOS_ID:
        break
      if char_id != PAD_ID and char_id != S_ID:
        f_row += [char_id]
    return delim.join(map(lambda x: vocab[x], [r for r in f_row if 0 < r < n]))
  else:
    return delim.join(map(lambda x: vocab[x], [r for r in row if 0 < r < n]))


def check_params(config, required_dict, optional_dict):
  if required_dict is None or optional_dict is None:
    return

  for pm, vals in required_dict.items():
    if pm not in config:
      raise ValueError("{} parameter has to be specified".format(pm))
    else:
      if vals and type(vals) is list and config[pm] not in vals:
        raise ValueError("{} has to be one of {}".format(pm, vals))
      if vals and type(vals) is not list and type(config[pm]) is not vals:
        raise ValueError("{} has to be of type {}".format(pm, vals))

  for pm, vals in optional_dict.items():
    if pm in config:
      if vals and type(vals) is list and config[pm] not in vals:
        raise ValueError("{} has to be one of {}".format(pm, vals))
      if vals and type(vals) is not list and type(config[pm]) is not vals:
        raise ValueError("{} has to be of type {}".format(pm, vals))

  for pm in config:
    if pm not in required_dict and pm not in optional_dict:
      raise ValueError("Unknown parameter: {}".format(pm))
