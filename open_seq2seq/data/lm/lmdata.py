# Copyright (c) 2017 NVIDIA Corporation
import random

import numpy as np
import tensorflow as tf
import os
from enum import Enum
from open_seq2seq.data.data_layer import DataLayer
from open_seq2seq.data.utils import load_pre_existing_vocabulary, pad_vocab_to_eight
from open_seq2seq.data.text2text.t2t import _read_and_batch_from_files

from open_seq2seq.data.lm.lmutils import Corpus

class LMTextDataLayer(DataLayer):
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      # 'content_file': str,
      # 'vocab_file': str,
      'shuffle': bool,
      'repeat': bool,
      'bptt': int,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'data_root': str,
      'rand_start': bool,
      'small': bool,
      'use_targets': bool,
      'delimiter': str,
      'target_file': str,
      'map_parallel_calls': int,
      'prefetch_buffer_size': int,
      'pad_lengths_to_eight': bool,
      'pad_vocab_to_eight': bool,
      'seed_tokens': str,
      'shuffle_buffer_size': int,
      'processed_data_folder': str,
    })

  def __init__(self, params, model, num_workers=1, worker_id=0):
    super(LMTextDataLayer, self).__init__(params, model,
                                          num_workers, worker_id)

    self._processed_data_folder = self.params.get('processed_data_folder', 'processed_data')
    self._data_root = self.params.get('data_root', None)
    self.corp = Corpus(self._data_root, self._processed_data_folder)
    if self.params['mode'] == 'train':
      self._batch_size = self.params['batch_size']
      self.corp.content = self.corp.train
    elif self.params['mode'] == 'eval':
      self._batch_size = self.params['batch_size']
      self.corp.content = self.corp.valid
    else:
      self._batch_size = 1
      self.corp.content = self.corp.test

    self.vocab_file = (self._processed_data_folder, 'vocab.txt')
    self.bptt = self.params['bptt']
    self.rand_start = self.params.get('rand_start', False)
    self._map_parallel_calls = self.params.get('map_parallel_calls', 8)
    self._pad_lengths_to_eight = self.params.get('pad_lengths_to_eight', False)
    self._prefetch_buffer_size = self.params.get('prefetch_buffer_size',
                                                 tf.contrib.data.AUTOTUNE)
    self._shuffle_buffer_size = self.params.get('shuffle_buffer_size', -1)
    self._num_workers = num_workers
    self._worker_id = worker_id
    self.params["delimiter"] = self.params.get("delimiter", " ")
    self.params["small"] = self.params.get("small", False)
    self.start = 0

    if self._pad_lengths_to_eight and not (self.params['max_length'] % 8 == 0):
      raise ValueError("If padding to 8 in data layer, then "
                       "max_length should be multiple of 8")

    # load source and target vocabularies to RAM

    seed_tokens = self.params.get('seed_tokens', 'The').split()
    
    self.params['end_token'] = self.corp.dictionary.word2idx[self.corp.dictionary.EOS]
    self.params['seed_tokens'] = [self.corp.dictionary.word2idx[seed_token] for seed_token in seed_tokens]
    if self.params["small"]:
      if self.params['mode'] == 'eval':
        self.corp.content = self.corp.content[:200]
      else:
        self.corp.content = self.corp.content[:9004]


    if self.params.get('pad_vocab_to_eight', False):
      self.corp.content = pad_vocab_to_eight(self.corp.content)

    if self.params['mode'] == 'infer':
      if len(self.corp.content) > self.bptt:
        self.corp.content = self.corp.content[-self.bptt:]

    self.dataset_size = len(self.corp.content)

    self.params['vocab_size'] = len(self.corp.dictionary.idx2word)
    self.PAD_ID = self.params['vocab_size']
    self.PAD = '<pad>'
    self.corp.dictionary.idx2word.append(self.PAD)
    self.corp.dictionary.word2idx[self.PAD] = self.PAD_ID

    self._input_tensors = {}
    self._batch_size

  def gen(self):
    while True:
      if self.rand_start:
        self.start = random.randint(0, self.bptt - 1)

      n_samples = (self.dataset_size - self.start - 1) // self.bptt

      for i in range(n_samples):
        begin = self.start + i * self.bptt
        yield (self.corp.content[begin : begin + self.bptt], self.corp.content[begin + 1 : begin + self.bptt + 1])

  def gen_infer(self):
    while True:
      yield (self.corp.content, self.corp.content)
    
  def build_graph(self):
    if self.params['mode'] == 'train' or self.params['mode'] == 'eval':
      gen = self.gen
      batch_shape = self.bptt
    else:
      gen = self.gen_infer
      batch_shape = len(self.corp.content)
    _src_tgt_dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32), 
                                (tf.TensorShape([batch_shape]), tf.TensorShape([batch_shape])))

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

    _src_tgt_dataset = _src_tgt_dataset.map(lambda x, y: ((x, tf.size(x)), (y, tf.size(y))), 
                            num_parallel_calls=self._map_parallel_calls)

    self.batched_dataset = _src_tgt_dataset.batch(self._batch_size)

    self._iterator = self.batched_dataset.make_initializable_iterator()

    if self.params['mode'] == 'train' or self.params['mode'] == 'eval':
      t1, t2 = self.iterator.get_next()
      x, x_length = t1[0], t1[1]
      y, y_length = t2[0], t2[1]
      self._input_tensors['source_tensors'] = [x, x_length]
      self._input_tensors['target_tensors'] = [y, y_length]
    else: # this is unncessary
      t1, _ = self.iterator.get_next()
      self._input_tensors['source_tensors'] = [t1[0], t1[1]]

  def get_size_in_samples(self):
    if self.params['mode'] == 'train' or self.params['mode'] == 'eval':
      return (self.dataset_size - self.start) // self.bptt
    return 1

  @property
  def iterator(self):
    return self._iterator

  @property
  def input_tensors(self):
    return self._input_tensors

class LMTextDataLayerGenerate(DataLayer):
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'vocab_file': str,
      'bptt': int,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'delimiter': str,
      'map_parallel_calls': int,
      'prefetch_buffer_size': int,
      'pad_lengths_to_eight': bool,
      'pad_vocab_to_eight': bool,
      'seed_file': str,
    })

  def __init__(self, params, model, num_workers=1, worker_id=0):
    super(LMTextDataLayerGenerate, self).__init__(params, model,
                                          num_workers, worker_id)
    self._batch_size = 1
    self.vocab_file = self.params['vocab_file']
    self.bptt = self.params['bptt']
    self._map_parallel_calls = self.params.get('map_parallel_calls', 8)
    self._pad_lengths_to_eight = self.params.get('pad_lengths_to_eight', False)
    self._prefetch_buffer_size = self.params.get('prefetch_buffer_size',
                                                 tf.contrib.data.AUTOTUNE)
    self._num_workers = num_workers
    self._worker_id = worker_id
    self.params["delimiter"] = self.params.get("delimiter", " ")
    self.seed_file = self.params.get("seed_file", None)

    # load source and target vocabularies to RAM
    self.corp = Corpus(self.params['vocab_file'])

    if self.seed_file:
      self.input_string = open(self.seed_file, 'r').read().strip()
    else:
      self.input_string = input('Please enter your seed string (case sensitive): ').strip()

    self.corp.content = self.corp.tokenize(self.input_string)
    if len(self.corp.content) > self.bptt:
      self.corp.content = self.corp.content[-self.bptt:]

    self.dataset_size = len(self.corp.content)

    self.params['vocab_size'] = len(self.corp.dictionary.idx2word)
    self.PAD_ID = self.params['vocab_size']
    self.PAD = '<pad>'
    self.corp.dictionary.idx2word.append(self.PAD)
    self.corp.dictionary.word2idx[self.PAD] = self.PAD_ID

    self._input_tensors = {}

  def gen(self):
    yield self.corp.content

  def build_graph(self):
    _src_tgt_dataset = tf.data.Dataset.from_generator(self.gen, (tf.int32), 
                                (tf.TensorShape([len(self.corp.content)])))

    _src_tgt_dataset = _src_tgt_dataset.map(lambda x: ((x, tf.size(x))), 
                            num_parallel_calls=self._map_parallel_calls)

    self.batched_dataset = _src_tgt_dataset.batch(self._batch_size)

    self._iterator = self.batched_dataset.make_initializable_iterator()
    t1, _ = self.iterator.get_next()
    t1 = tf.expand_dims(t1[0], axis=0)
    print(t1)
    self._input_tensors['source_tensors'] = [t1[0], t1[1]]
    print(self._input_tensors)


  def get_size_in_samples(self):
    return 1

  @property
  def iterator(self):
    return self._iterator

  @property
  def input_tensors(self):
    return self._input_tensors