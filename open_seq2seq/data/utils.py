# Copyright (c) 2017 NVIDIA Corporation
import random
from . import data_layer
import re
import copy
import nltk

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

def calculate_bleu(preds, targets):
  '''
  :param preds: list of lists
  :param targets: list of lists
  :return: bleu score - float32
  '''
  bleu_score = nltk.translate.bleu_score.corpus_bleu(targets, preds, emulate_multibleu=True)
  print("EVAL BLEU")
  print(bleu_score)
  return bleu_score

def deco_print(line):
  print(">==================> " + line)

def configure_params(in_config, mode="train"):
  config = copy.deepcopy(in_config)
  config["mode"] = mode
  if mode == "infer":
    config["shuffle"] = False
    config["encoder_dp_input_keep_prob"] = 1.0
    config["decoder_dp_input_keep_prob"] = 1.0
    config["batch_size"] = 1
    config["num_gpus"] = 1
    config["source_file"] = config["source_file_test"]
    config["target_file"] = config["target_file_test"]
    if "bucket_src_test" in config:
      config["bucket_src"] = config["bucket_src_test"]
    if "bucket_tgt_test" in config:
      config["bucket_tgt"] = config["bucket_tgt_test"]
  elif mode == "eval":
    config["mode"] = "train" # this is for dl, to output ys
    config['source_file'] = config['source_file_eval']
    config['target_file'] = config['target_file_eval']
    config["shuffle"] = False
    config["encoder_dp_input_keep_prob"] = 1.0
    config["decoder_dp_input_keep_prob"] = 1.0
    config["num_gpus"] = 1
    if "length_penalty" in config:
      # this is needed to have beam search on GPU
      # see: https://github.com/tensorflow/nmt/issues/110 waiting for tf
      # might be fixed in TF r1.4
      deco_print("INFO: In Eval model, beam search length penalty was set to 0")
      config["length_penalty"] = 0.0
    if "bucket_src_test" in config:
      config["bucket_src"] = config["bucket_src_test"]
    if "bucket_tgt_test" in config:
      config["bucket_tgt"] = config["bucket_tgt_test"]
  elif mode == "train":
    config["shuffle"] = True
    config["decoder_type"] = "greedy"
  else:
    raise ValueError("Unknown mode")
  return config