# Copyright (c) 2019 NVIDIA Corporation
"""This file has a CTC Greedy decoder"""
import numpy as np

def ctc_greedy_decoder(logits, wordmap, step_size,
                       blank_idx, start_shift, end_shift):
  """Decodes logits to chars using greedy ctc format,
  outputs start and end time for every word
  Args :
    logits: time x chars (log probabilities)
    wordmap: vocab (maps index to characters)
    step_size: number of steps to take in time domain per block of input
    blank_idx: index of blank char
    start_shift: shift needed for start of word in time domain
    end_shift: shift needed in end of word in time domain
  """
  prev_idx = -1
  output = []
  start = []
  end = []
  lst_letter = -1
  for i, log_prob in enumerate(logits):
    idx = np.argmax(log_prob)
    if idx not in (blank_idx, prev_idx):
      if len(output) == 0:
        start.append(step_size*i+start_shift)
      else:
        if output[-1] == " ":
          start.append(max(step_size*i+start_shift, end[-1]))
      output += wordmap[idx]
      if output[-1] == " ":
        end.append(step_size*lst_letter+end_shift)
      lst_letter = i
    prev_idx = idx
  end.append(step_size*lst_letter+end_shift)
  output = "".join(output)
  output = output.strip(" ")
  return output, start, end
