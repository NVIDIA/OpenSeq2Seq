# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import os
from enum import Enum
from open_seq2seq.data.data_layer import DataLayer
from open_seq2seq.data.utils import load_pre_existing_vocabulary, pad_vocab_to_eight
from open_seq2seq.data.text2text.t2t import _read_and_batch_from_files
from open_seq2seq.data.text2text.tokenizer import PAD_ID


class SpecialTextTokens(Enum):
  PAD_ID = 0  # special padding token
  EOS_ID = 1  # special end of sentence token
  S_ID = 2  # special start of sentence token
  UNK_ID = 3  # out-of-vocabulary tokens will map there
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
      'pad_vocab_to_eight': bool,
      'shuffle_buffer_size': int,
    })

  def __init__(self, params, model, num_workers=1, worker_id=0):
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
    self._prefetch_buffer_size = self.params.get('prefetch_buffer_size',
                                                 tf.contrib.data.AUTOTUNE)
    self._shuffle_buffer_size = self.params.get('shuffle_buffer_size', -1)
    self._num_workers = num_workers
    self._worker_id = worker_id
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
      min_idx=SpecialTextTokens.UNK_ID.value + 1)
    self.tgt_seq2idx = load_pre_existing_vocabulary(
      self.tgt_vocab_file,
      min_idx=SpecialTextTokens.UNK_ID.value + 1)

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

    self._input_tensors = {}

  def build_graph(self):
    if self.params["mode"] == "interactive_infer":
      return self._build_interactive_graph()

    def pad2eight(lst, do_pad_eight):
      if len(lst) % 8 == 0 or not do_pad_eight:
        return lst
      else:
        return lst + [SpecialTextTokens.PAD_ID.value] * (8 - len(lst) % 8)

    def src_token_to_id(line):
      tokens = line.decode("utf-8").split(self._delimiter)
      return np.array(pad2eight([SpecialTextTokens.S_ID.value] + \
             [self.src_seq2idx.get(token, SpecialTextTokens.UNK_ID.value) for token in tokens[:self.max_len-2]] + \
             [SpecialTextTokens.EOS_ID.value], self._pad_lengths_to_eight), dtype="int32")

    def tgt_token_to_id(line):
      tokens = line.decode("utf-8").split(self._delimiter)
      return np.array(pad2eight([SpecialTextTokens.S_ID.value] + \
             [self.tgt_seq2idx.get(token, SpecialTextTokens.UNK_ID.value) for token in tokens[:self.max_len-2]] + \
             [SpecialTextTokens.EOS_ID.value], self._pad_lengths_to_eight), dtype="int32")

    _sources = tf.data.TextLineDataset(self.source_file)\
      .map(lambda line: tf.py_func(func=src_token_to_id, inp=[line],
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
    ).cache()

    if self._num_workers > 1:
      _src_tgt_dataset = _src_tgt_dataset\
        .shard(num_shards=self._num_workers, index=self._worker_id)

    if self.params['shuffle']:
      bf_size = self.get_size_in_samples() if self._shuffle_buffer_size == -1 \
                                           else self._shuffle_buffer_size
      _src_tgt_dataset = _src_tgt_dataset.shuffle(buffer_size=bf_size)
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

    self._iterator = self.batched_dataset.make_initializable_iterator()

    if self.params['mode'] == 'train' or self.params['mode'] == 'eval':
      t1, t2 = self.iterator.get_next()
      x, x_length = t1[0], t1[1]
      y, y_length = t2[0], t2[1]
      self._input_tensors['source_tensors'] = [x, x_length]
      self._input_tensors['target_tensors'] = [y, y_length]
    else:
      t1, _ = self.iterator.get_next()
      self._input_tensors['source_tensors'] = [t1[0], t1[1]]

  def _build_interactive_graph(self):
    """
    Must pass in placeholder
    """
    # def pad2eight(lst, do_pad_eight):
    #   if len(lst) % 8 == 0 or not do_pad_eight:
    #     return lst
    #   else:
    #     return lst + [SpecialTextTokens.PAD_ID.value] * (8 - len(lst) % 8)
    #
    # def src_token_to_id(line):
    #   tokens = line.decode("utf-8").split(self._delimiter)
    #   return np.array(pad2eight([SpecialTextTokens.S_ID.value] + \
    #          [self.src_seq2idx.get(token, SpecialTextTokens.UNK_ID.value) for token in tokens[:self.max_len-2]] + \
    #          [SpecialTextTokens.EOS_ID.value], self._pad_lengths_to_eight), dtype="int32")
    #
    # def tgt_token_to_id(line):
    #   tokens = line.decode("utf-8").split(self._delimiter)
    #   return np.array(pad2eight([SpecialTextTokens.S_ID.value] + \
    #          [self.tgt_seq2idx.get(token, SpecialTextTokens.UNK_ID.value) for token in tokens[:self.max_len-2]] + \
    #          [SpecialTextTokens.EOS_ID.value], self._pad_lengths_to_eight), dtype="int32")
    #
    # self.input = tf.placeholder(dtype=tf.string)
    # self._dataset = tf.data.Dataset.from_tensor_slices(
    #   self.input
    # )
    # self._dataset = self._dataset \
    #   .map(lambda line: tf.py_func(func=src_token_to_id, inp=[line],
    #                                Tout=[tf.int32], stateful=False),
    #        num_parallel_calls=self._map_parallel_calls) \
    #   .map(lambda tokens: (tokens, tf.size(tokens)),
    #        num_parallel_calls=self._map_parallel_calls)
    #
    # self._dataset = tf.data.Dataset.zip((self._dataset, self._dataset))
    #
    # self.batched_dataset = self._dataset.padded_batch(
    #   self._batch_size,
    #   padded_shapes=((tf.TensorShape([None]),
    #                   tf.TensorShape([])),
    #                  (tf.TensorShape([None]),
    #                   tf.TensorShape([]))),
    #   padding_values=(
    #     (SpecialTextTokens.PAD_ID.value,
    #      0),
    #     (SpecialTextTokens.PAD_ID.value,
    #      0))).prefetch(tf.contrib.data.AUTOTUNE)
    #
    # self._iterator = self._dataset.make_initializable_iterator()
    #
    # t1, _ = self.iterator.get_next()
    # self._input_tensors['source_tensors'] = [t1[0], t1[1]]


    def pad2eight(lst, do_pad_eight):
      if len(lst) % 8 == 0 or not do_pad_eight:
        return lst
      else:
        return lst + [SpecialTextTokens.PAD_ID.value] * (8 - len(lst) % 8)

    def src_token_to_id(line):
      tokens = line.decode("utf-8").split(self._delimiter)
      return np.array(pad2eight([SpecialTextTokens.S_ID.value] + \
             [self.src_seq2idx.get(token, SpecialTextTokens.UNK_ID.value) for token in tokens[:self.max_len-2]] + \
             [SpecialTextTokens.EOS_ID.value], self._pad_lengths_to_eight), dtype="int32")

    def tgt_token_to_id(line):
      tokens = line.decode("utf-8").split(self._delimiter)
      return np.array(pad2eight([SpecialTextTokens.S_ID.value] + \
             [self.tgt_seq2idx.get(token, SpecialTextTokens.UNK_ID.value) for token in tokens[:self.max_len-2]] + \
             [SpecialTextTokens.EOS_ID.value], self._pad_lengths_to_eight), dtype="int32")

    self.input = tf.placeholder(dtype=tf.string)

    _sources = tf.data.Dataset.from_tensor_slices(self.input)\
      .map(lambda line: tf.py_func(func=src_token_to_id, inp=[line],
                                   Tout=[tf.int32], stateful=False),
           num_parallel_calls=self._map_parallel_calls) \
      .map(lambda tokens: (tokens, tf.size(tokens)),
           num_parallel_calls=self._map_parallel_calls)

    _targets = tf.data.Dataset.from_tensor_slices(self.input) \
      .map(lambda line: tf.py_func(func=tgt_token_to_id, inp=[line],
                                   Tout=[tf.int32], stateful=False),
           num_parallel_calls=self._map_parallel_calls) \
      .map(lambda tokens: (tokens, tf.size(tokens)),
           num_parallel_calls=self._map_parallel_calls)

    _src_tgt_dataset = tf.data.Dataset.zip((_sources, _targets)).filter(
      lambda t1, t2: tf.logical_and(tf.less_equal(t1[1], self.max_len),
                                    tf.less_equal(t2[1], self.max_len))
    ).cache()

    if self._num_workers > 1:
      _src_tgt_dataset = _src_tgt_dataset\
        .shard(num_shards=self._num_workers, index=self._worker_id)

    if self.params['shuffle']:
      bf_size = self.get_size_in_samples() if self._shuffle_buffer_size == -1 \
                                           else self._shuffle_buffer_size
      _src_tgt_dataset = _src_tgt_dataset.shuffle(buffer_size=bf_size)
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

    self._iterator = self.batched_dataset.make_initializable_iterator()

    if self.params['mode'] == 'train' or self.params['mode'] == 'eval':
      t1, t2 = self.iterator.get_next()
      x, x_length = t1[0], t1[1]
      y, y_length = t2[0], t2[1]
      self._input_tensors['source_tensors'] = [x, x_length]
      self._input_tensors['target_tensors'] = [y, y_length]
    else:
      t1, _ = self.iterator.get_next()
      self._input_tensors['source_tensors'] = [t1[0], t1[1]]

  def get_size_in_samples(self):
    if self.params["mode"] == "interactive_infer":
      return 1
    return self.dataset_size

  @property
  def iterator(self):
    return self._iterator

  @property
  def input_tensors(self):
    return self._input_tensors

class TransformerDataLayer(DataLayer):
  """Wraps Transformers data pipeline into the form for OpenSeq2Seq"""
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'data_dir': str,
      'file_pattern': str,
      'src_vocab_file': str,
      'batch_size': int,
      'max_length': int,
      'shuffle': bool,
      "delimiter": str,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'repeat': int,
      'num_cpu_cores': int,
      'tgt_vocab_file': str,
      'pad_data_to_eight': bool,
      'batch_in_tokens': bool,
    })

  def __init__(self, params, model, num_workers=1, worker_id=0):
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

    self._num_workers = num_workers
    self._worker_id = worker_id

    self._input_tensors = {}
    self._iterator = None
    self.batched_dataset = None

  def build_graph(self):
    file_pattern = os.path.join(self.params['data_dir'],
                                self.params['file_pattern'])
    self.batched_dataset = _read_and_batch_from_files(
      file_pattern=file_pattern,
      batch_size=self.params['batch_size'],
      max_length=self.params['max_length'],
      num_cpu_cores=self.params.get('num_cpu_cores', 2),
      shuffle=self.params['shuffle'],
      repeat=self.params['repeat'],
      num_workers=self._num_workers,
      worker_id=self._worker_id,
      batch_in_tokens=self.params.get('batch_in_tokens', True),
      pad2eight=self.params.get('pad_data_to_eight', False))

    self._iterator = self.batched_dataset.make_initializable_iterator()
    x, y = self.iterator.get_next()

    len_x = tf.count_nonzero(x, axis=1, dtype=tf.int32)
    len_y = tf.count_nonzero(y, axis=1, dtype=tf.int32)
    if self.params['mode'] == 'train' or self.params['mode'] == 'eval':
      self._input_tensors['source_tensors'] = [x, len_x]
      self._input_tensors['target_tensors'] = [y, len_y]
    else:
      self._input_tensors['source_tensors'] = [x, len_x]

  @property
  def iterator(self):
    return self._iterator

  @property
  def input_tensors(self):
    return self._input_tensors
