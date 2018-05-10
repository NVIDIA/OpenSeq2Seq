# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import numpy as np
import tensorflow as tf
import random
import copy
import io
import os
from enum import Enum
from open_seq2seq.data.data_layer import DataLayer
from open_seq2seq.data.utils import load_pre_existing_vocabulary, pad_vocab_to_eight
from open_seq2seq.data.text2text.t2t import _read_and_batch_from_files
from open_seq2seq.data.text2text.tokenizer import PAD_ID, PAD, EOS_ID, EOS
class SpecialTextTokens(Enum):
  UNK_ID = 0  # out-of-vocabulary tokens will map there
  S_ID = 1  # special start of sentence token
  EOS_ID = 2  # special end of sentence token
  PAD_ID = 3  # special padding token
  OUT_OF_BUCKET = 1234567890
  END_OF_CHOICE = -100

  @staticmethod
  def to_string(s_token):
    if s_token == SpecialTextTokens.UNK_ID.value:
      return '<UNK>'
    elif s_token == SpecialTextTokens.S_ID.value:
      return '<S>'
    elif s_token == SpecialTextTokens.EOS_ID.value:
      return '</S>'
    elif s_token == SpecialTextTokens.PAD_ID.value:
      return '<PAD>'
    else:
      raise ValueError("Unknown Value in SpecialTokens")


def weighted_choice(choices):
  total_weights = sum(w for c, w in choices.items())
  if total_weights <= 0:
    return SpecialTextTokens.END_OF_CHOICE.value
  r = random.uniform(0, total_weights)
  mx = 0
  for i, w in choices.items():
    if mx + w >= r:
      return i
    mx += w
  raise AssertionError("weighted choice got to the wrong place")

class ParallelTextDataLayer(DataLayer):
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'source_file': str,
      'src_vocab_file': str,
      'tgt_vocab_file': str,
      'max_length': int,
      'shuffle': bool,
      'repeat': bool,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'use_targets': bool,
      'delimiter': str,
      'target_file': str,
      'map_parallel_calls': int,
      'prefetch_buffer_size': int,
      'pad_lengths_to_eight': bool,
      'pad_vocab_to_eight' : bool,
    })

  def __init__(self, params, model, num_workers=None, worker_id=None):
    super(ParallelTextDataLayer, self).__init__(params, model,
                                                num_workers, worker_id)
    self._batch_size = self.params['batch_size']
    self.source_file = self.params['source_file']
    self._use_targets = self.params.get('use_targets', True)
    if not self._use_targets:
      self.target_file = self.source_file
      if 'target_file' in self.params:
        print("WARNING: target file was specified but was "
              "ignored by data layer because 'use_targets'=False")
    else:
      self.target_file = self.params['target_file']
    self.src_vocab_file = self.params['src_vocab_file']
    self.tgt_vocab_file = self.params['tgt_vocab_file']
    self.max_len = self.params['max_length']
    self._delimiter = self.params.get('delimiter', ' ')
    self._map_parallel_calls = self.params.get('map_parallel_calls', 8)
    self._pad_lengths_to_eight = self.params.get('pad_lengths_to_eight', False)
    self._prefetch_buffer_size = self.params.get('prefetch_buffer_size', 4)
    if self._pad_lengths_to_eight and not (self.params['max_length'] % 8 == 0):
      raise ValueError("If padding to 8 in data layer, then "
                       "max_length should be multiple of 8")

    def file_len(fname):
      with open(fname) as f:
        for i, l in enumerate(f):
          pass
      return i + 1

    self.dataset_size = file_len(self.source_file)

    # load source and target vocabularies to RAM
    self.src_seq2idx = load_pre_existing_vocabulary(
      self.src_vocab_file,
      min_idx=SpecialTextTokens.PAD_ID.value + 1)
    self.tgt_seq2idx = load_pre_existing_vocabulary(
      self.tgt_vocab_file,
      min_idx=SpecialTextTokens.PAD_ID.value + 1)

    # unknown symbol
    self.src_seq2idx[
      SpecialTextTokens.to_string(SpecialTextTokens.UNK_ID.value)] = \
      SpecialTextTokens.UNK_ID.value
    self.tgt_seq2idx[
      SpecialTextTokens.to_string(SpecialTextTokens.UNK_ID.value)] = \
      SpecialTextTokens.UNK_ID.value

    # sentence start
    self.src_seq2idx[
      SpecialTextTokens.to_string(SpecialTextTokens.S_ID.value)] = \
      SpecialTextTokens.S_ID.value
    self.tgt_seq2idx[
      SpecialTextTokens.to_string(SpecialTextTokens.S_ID.value)] = \
      SpecialTextTokens.S_ID.value
    # sentence end
    self.src_seq2idx[
      SpecialTextTokens.to_string(SpecialTextTokens.EOS_ID.value)] = \
      SpecialTextTokens.EOS_ID.value
    self.tgt_seq2idx[
      SpecialTextTokens.to_string(SpecialTextTokens.EOS_ID.value)] = \
      SpecialTextTokens.EOS_ID.value
    # padding
    self.src_seq2idx[
      SpecialTextTokens.to_string(SpecialTextTokens.PAD_ID.value)] = \
      SpecialTextTokens.PAD_ID.value
    self.tgt_seq2idx[
      SpecialTextTokens.to_string(SpecialTextTokens.PAD_ID.value)] = \
      SpecialTextTokens.PAD_ID.value

    if self.params.get('pad_vocab_to_eight', False):
      self.src_seq2idx = pad_vocab_to_eight(self.src_seq2idx)
      self.tgt_seq2idx = pad_vocab_to_eight(self.tgt_seq2idx)

    self.src_idx2seq = {idx: w for w, idx in self.src_seq2idx.items()}
    self.tgt_idx2seq = {idx: w for w, idx in self.tgt_seq2idx.items()}

    self.params['src_vocab_size'] = len(self.src_seq2idx)
    self.params['tgt_vocab_size'] = len(self.tgt_seq2idx)
    self.params['target_seq2idx'] = self.tgt_seq2idx
    self.params['source_seq2idx'] = self.src_seq2idx
    self.params['target_idx2seq'] = self.tgt_idx2seq
    self.params['source_idx2seq'] = self.src_idx2seq

  def build_graph(self):
    def pad2eight(lst, do_pad_eight):
      if len(lst) % 8 == 0 or not do_pad_eight:
        return lst
      else:
        return lst + [SpecialTextTokens.PAD_ID.value] * (8 - len(lst) % 8)

    def src_token_to_id(line):
      tokens = line.decode("utf-8").split(self._delimiter)
      return np.array(pad2eight([SpecialTextTokens.S_ID.value] + \
             [self.src_seq2idx.get(token, SpecialTextTokens.UNK_ID.value) for token in tokens] + \
             [SpecialTextTokens.EOS_ID.value], self._pad_lengths_to_eight), dtype="int32")

    def tgt_token_to_id(line):
      tokens = line.decode("utf-8").split(self._delimiter)
      return np.array(pad2eight([SpecialTextTokens.S_ID.value] + \
             [self.tgt_seq2idx.get(token, SpecialTextTokens.UNK_ID.value) for token in tokens] + \
             [SpecialTextTokens.EOS_ID.value], self._pad_lengths_to_eight), dtype="int32")

    _sources = tf.data.TextLineDataset(self.source_file)\
      .map(lambda line: tf.py_func(func=src_token_to_id,inp=[line],
                                   Tout=[tf.int32], stateful=False),
           num_parallel_calls=self._map_parallel_calls) \
      .map(lambda tokens: (tokens, tf.size(tokens)),
           num_parallel_calls=self._map_parallel_calls)

    _targets = tf.data.TextLineDataset(self.target_file) \
      .map(lambda line: tf.py_func(func=tgt_token_to_id, inp=[line],
                                   Tout=[tf.int32], stateful=False),
           num_parallel_calls=self._map_parallel_calls) \
      .map(lambda tokens: (tokens, tf.size(tokens)),
           num_parallel_calls=self._map_parallel_calls)

    _src_tgt_dataset = tf.data.Dataset.zip((_sources, _targets)).filter(
      lambda t1, t2: tf.logical_and(tf.less_equal(t1[1], self.max_len),
                                    tf.less_equal(t2[1], self.max_len))
    )

    if self.params['shuffle']:
      _src_tgt_dataset = _src_tgt_dataset\
        .shuffle(buffer_size=self.get_size_in_samples())
    else:
      _src_tgt_dataset = _src_tgt_dataset

    if self.params['repeat']:
      _src_tgt_dataset = _src_tgt_dataset.repeat()

    self.batched_dataset = _src_tgt_dataset.padded_batch(
      self._batch_size,
      padded_shapes=((tf.TensorShape([None]),
                      tf.TensorShape([])),
                     (tf.TensorShape([None]),
                      tf.TensorShape([]))),
      padding_values=(
      (SpecialTextTokens.PAD_ID.value,
       0),
      (SpecialTextTokens.PAD_ID.value,
       0))).prefetch(buffer_size=self._prefetch_buffer_size)

    self.iterator = self.batched_dataset.make_one_shot_iterator()

  def gen_input_tensors(self):
    if self._use_targets:
      t1, t2 = self.iterator.get_next()
      x, x_length = t1[0], t1[1]
      y, y_length = t2[0], t2[1]
      return [x, x_length, y, y_length]
    else:
      t1, _ = self.iterator.get_next()
      return [t1[0], t1[1]]

  def next_batch_feed_dict(self):
    return {}

  def shuffle(self):
    pass

  def get_size_in_samples(self):
    return self.dataset_size

class TransformerDataLayer(DataLayer):
  """Wraps Transformers data pipeline into the form for OpenSeq2Seq"""

  ################################
  # ------------------------------
  # Get rid of these
  # ------------------------------
  def next_batch_feed_dict(self):
    pass

  def shuffle(self):
    pass

  def get_size_in_samples(self):
    pass
  ################################

  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'data_dir': str,
      'file_pattern': str,
      'src_vocab_file': str,
      'batch_size': int,
      'max_length': int,
      'shuffle': bool,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'repeat': int,
      'num_cpu_cores': int,
      'tgt_vocab_file': str,
    })

  def __init__(self, params, model, num_workers=None, worker_id=None):
    super(TransformerDataLayer, self).__init__(params, model,
                                                      num_workers, worker_id)
    self.src_vocab_file = self.params['src_vocab_file']
    # if tgt vocab isn't specified - assume common vocab file
    self.tgt_vocab_file = self.params.get('tgt_vocab_file', self.src_vocab_file)

    # load source and target vocabularies to RAM
    # pre-processed vocab starts from PAD, EOS
    self.src_seq2idx = load_pre_existing_vocabulary(
      self.src_vocab_file,
      min_idx=PAD_ID)
    self.tgt_seq2idx = load_pre_existing_vocabulary(
      self.tgt_vocab_file,
      min_idx=PAD_ID)

    self.src_idx2seq = {idx: w for w, idx in self.src_seq2idx.items()}
    self.tgt_idx2seq = {idx: w for w, idx in self.tgt_seq2idx.items()}

    self.params['src_vocab_size'] = len(self.src_seq2idx)
    self.params['tgt_vocab_size'] = len(self.tgt_seq2idx)
    self.params['target_seq2idx'] = self.tgt_seq2idx
    self.params['source_seq2idx'] = self.src_seq2idx
    self.params['target_idx2seq'] = self.tgt_idx2seq
    self.params['source_idx2seq'] = self.src_idx2seq

  def build_graph(self):
    file_pattern = os.path.join(self.params['data_dir'],
                                self.params['file_pattern'])
    self.batched_dataset = _read_and_batch_from_files(
      file_pattern=file_pattern,
      batch_size=self.params['batch_size'],
      max_length=self.params['max_length'],
      num_cpu_cores=self.params.get('num_cpu_cores', 2),
      shuffle=self.params['shuffle'],
      repeat=1)

    #self.iterator = self.batched_dataset.make_one_shot_iterator()
    self.iterator = self.batched_dataset.make_initializable_iterator()

  def get_iterator(self):
    return self.iterator

  #def redo_iterator(self):
  #  self.iterator = self.batched_dataset.make_one_shot_iterator()

  def gen_input_tensors(self):
    if self._input_tensors is None:
      x, y = self.iterator.get_next()
      len_x = tf.count_nonzero(x, axis=1, dtype=tf.int32)
      len_y = tf.count_nonzero(y, axis=1, dtype=tf.int32)
      self._input_tensors = x, len_x, y, len_y
    else:
      print("----->>> WARNING: Attempting to generate existing input tensors")
    return tuple(self._input_tensors)








