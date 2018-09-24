# Copyright (c) 2018 NVIDIA Corporation
import random

import numpy as np
import tensorflow as tf
import os
from enum import Enum
from open_seq2seq.data.data_layer import DataLayer
from open_seq2seq.data.utils import load_pre_existing_vocabulary, pad_vocab_to_eight
from open_seq2seq.data.text2text.t2t import _read_and_batch_from_files

from open_seq2seq.data.lm.lmutils import Dictionary, Corpus, IMDBCorpus, SSTCorpus

class WKTDataLayer(DataLayer):
  '''
  WKTDataLayer does the necessary pre-processing to make the WikiText datasets 
  ready to be fed into the model. We use the ``word_token`` method 
  available in the ``nltk`` package. 
  You can download the datasets here:
  https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
  bptt: backpropagation through time - the length of the sequences used for training
  rand_start: whether to start from a random starting index between (0, bptt)
  '''
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
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
      'map_parallel_calls': int,
      'prefetch_buffer_size': int,
      'pad_lengths_to_eight': bool,
      'pad_vocab_to_eight': bool,
      'seed_tokens': str,
      'shuffle_buffer_size': int,
      'processed_data_folder': str,
    })

  def __init__(self, params, model, num_workers=1, worker_id=0):
    super(WKTDataLayer, self).__init__(params, model,
                                          num_workers, worker_id)

    self._processed_data_folder = self.params.get('processed_data_folder', 'wkt-processed_data')
    self._data_root = self.params.get('data_root', None)

    self.corp = Corpus(self._data_root, self._processed_data_folder)

    seed_tokens = self.params.get('seed_tokens', 'The').split()
    
    self.end_token = self.corp.dictionary.word2idx[self.corp.dictionary.EOS]
    self.params['seed_tokens'] = [self.corp.dictionary.word2idx[seed_token] for seed_token in seed_tokens]
    
    if self.params['mode'] == 'infer':
      self.corp.content = self.params['seed_tokens']

    if self.params['mode'] == 'train':
      self.batch_size = self.params['batch_size']
      self.corp.content = self.corp.train
    elif self.params['mode'] == 'eval':
      self.batch_size = self.params['batch_size']
      self.corp.content = self.corp.valid
    else:
      if len(self.corp.content) < self.params['batch_size']:
        self.batch_size = len(self.corp.content)
      else:
        self.batch_size = self.params['batch_size']

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
    self.delimiter = self.params.get("delimiter", " ")
    self._small = self.params.get("small", False)
    self.start = 0

    # load source and target vocabularies to RAM
    if self._small:
      if self.params['mode'] == 'eval':
        self.corp.content = self.corp.content[:200]
      else:
        self.corp.content = self.corp.content[:9004]

    if self.params.get('pad_vocab_to_eight', False):
      self.corp.content = pad_vocab_to_eight(self.corp.content)

    self.dataset_size = len(self.corp.content)
    self.vocab_size = len(self.corp.dictionary.idx2word)
    self._input_tensors = {}

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
      for seed in self.corp.content:
        yield ([seed], [seed])
      
  def build_graph(self):
    if self.params['mode'] == 'train' or self.params['mode'] == 'eval':
      gen = self.gen
      batch_shape = self.bptt
    else:
      gen = self.gen_infer
      batch_shape = 1
    
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

    self.batched_dataset = _src_tgt_dataset.batch(self.batch_size)

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
    return len(self.corp.content)

  @property
  def iterator(self):
    return self._iterator

  @property
  def input_tensors(self):
    return self._input_tensors

class TextClassificationDataLayer(DataLayer):
  '''
  The base ckass to process data for text classification tasks.
  If the data has already been processed, it shoud load the processed
  data instead of re-processing it.
  '''
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'lm_vocab_file': str,
      'shuffle': bool,
      'repeat': bool,
      'max_length': int,
      'processed_data_folder': str,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'rand_start': bool,
      'small': bool,
      'use_targets': bool,
      'delimiter': str,
      'map_parallel_calls': int,
      'prefetch_buffer_size': int,
      'pad_lengths_to_eight': bool,
      'pad_vocab_to_eight': bool,
      'shuffle_buffer_size': int,
      'data_root': str,
      'binary': bool,
      'num_classes': int,
      'get_stats': bool,
    })

  def __init__(self, params, model, num_workers=1, worker_id=0):
    super(TextClassificationDataLayer, self).__init__(params, model,
                                          num_workers, worker_id)

    self._data_root = self.params.get('data_root', None)
    self._binary = self.params.get('binary', True)
    self._get_stats = self.params.get('get_stats', False)
    self._lm_vocab_file = self.params['lm_vocab_file']

    self._map_parallel_calls = self.params.get('map_parallel_calls', 8)
    self._pad_lengths_to_eight = self.params.get('pad_lengths_to_eight', False)
    self._prefetch_buffer_size = self.params.get('prefetch_buffer_size',
                                                 tf.contrib.data.AUTOTUNE)
    self._shuffle_buffer_size = self.params.get('shuffle_buffer_size', -1)
    self._num_workers = num_workers
    self._worker_id = worker_id
    self._small = self.params.get("small", False)
    self._max_length = self.params['max_length']
    self.delimiter = self.params.get("delimiter", " ")
    self.EOS_ID = -1
    self.batch_size = self.params['batch_size']

    if self._pad_lengths_to_eight and not (self._max_length % 8 == 0):
      raise ValueError("If padding to 8 in data layer, then "
                       "max_length should be multiple of 8")
    self._input_tensors = {}

  def gen(self):
    while True:
      for review, raw_rating in self.corp.content:
        if len(review) > self._max_length:
          review = review[-self._max_length:]
        rating = np.zeros(self.num_classes)
        rating[raw_rating] = 1
        yield (review, rating)

  def build_graph(self):
    _src_tgt_dataset = tf.data.Dataset.from_generator(self.gen, 
                                        (tf.int32, tf.int32), 
                                        (tf.TensorShape([None]), tf.TensorShape([self.num_classes])))

    if self._num_workers > 1:
      _src_tgt_dataset = _src_tgt_dataset\
        .shard(num_shards=self._num_workers, index=self._worker_id)

    if self.params['shuffle']:
      bf_size = self.get_size_in_samples() if self._shuffle_buffer_size == -1 \
                                           else self._shuffle_buffer_size
      _src_tgt_dataset = _src_tgt_dataset.shuffle(buffer_size=bf_size)

    if self.params['repeat']:
      _src_tgt_dataset = _src_tgt_dataset.repeat()

    _src_tgt_dataset = _src_tgt_dataset.map(lambda x, y: ((x, tf.size(x)), (y, tf.size(y))), 
                            num_parallel_calls=self._map_parallel_calls)

    self.batched_dataset = _src_tgt_dataset.padded_batch(
      self.batch_size,
      padded_shapes=((tf.TensorShape([None]),
                      tf.TensorShape([])),
                     (tf.TensorShape([None]),
                      tf.TensorShape([]))),
      padding_values=(
      (self.EOS_ID, 0),
      (self.EOS_ID, 0))).prefetch(buffer_size=self._prefetch_buffer_size)

    self._iterator = self.batched_dataset.make_initializable_iterator()

    t1, t2 = self.iterator.get_next()
    x, x_length = t1[0], t1[1]
    y, y_length = t2[0], t2[1]
    self._input_tensors['source_tensors'] = [x, x_length]
    self._input_tensors['target_tensors'] = [y, y_length]

  def get_size_in_samples(self):
    return self.dataset_size

  @property
  def iterator(self):
    return self._iterator

  @property
  def input_tensors(self):
    return self._input_tensors

class IMDBDataLayer(TextClassificationDataLayer):
  '''
  Data layer to process the raw IMDB data, which can be downloaded here:
  http://ai.stanford.edu/~amaas/data/sentiment/

  '''
  def __init__(self, params, model, num_workers=1, worker_id=0):
    super(IMDBDataLayer, self).__init__(params, model, num_workers, worker_id)
    self._processed_data_folder = self.params['processed_data_folder']

    if self._binary:
      self.num_classes = 2
    else:
      self.num_classes = 10

    self.corp = IMDBCorpus(self._data_root, 
                          self._processed_data_folder, 
                          self._lm_vocab_file, 
                          self._binary,
                          get_stats=self._get_stats)
    
    if self.params['mode'] == 'train':
      self.corp.content = self.corp.train
    elif self.params['mode'] == 'eval':
      self.corp.content = self.corp.valid
    else:
      self.corp.content = self.corp.test    
    
    if self._small:
      if self.params['mode'] == 'eval':
        self.corp.content = self.corp.content[:self.batch_size * 2]
      else:
        self.corp.content = self.corp.content[:self.batch_size * 4]

    self.dataset_size = len(self.corp.content)
    self.vocab_size = len(self.corp.dictionary.idx2word)
    self.EOS_ID = self.corp.dictionary.word2idx[self.corp.dictionary.EOS]
    self.end_token = self.corp.dictionary.word2idx[self.corp.dictionary.EOS]

class SSTDataLayer(TextClassificationDataLayer):
  '''
  Data layer to process the raw SST (Stanford Sentiment Treebank).
  Read about the dataset here:
  https://nlp.stanford.edu/sentiment/
  Download the preprocessed version that can be used for this DataLayer here:
  https://github.com/NVIDIA/sentiment-discovery/tree/master/data/binary_sst
  '''
  def __init__(self, params, model, num_workers=1, worker_id=0):
    super(SSTDataLayer, self).__init__(params, model, num_workers, worker_id)
    self._processed_data_folder = self.params['processed_data_folder']
    self.corp = SSTCorpus(self._data_root, 
                          self._processed_data_folder, 
                          self._lm_vocab_file,
                          get_stats=self._get_stats)
    
    if self.params['mode'] == 'train':
      self.corp.content = self.corp.train
    elif self.params['mode'] == 'eval':
      self.corp.content = self.corp.valid
    else:
      self.corp.content = self.corp.test
    self.num_classes = 2
    self.dataset_size = len(self.corp.content)
    self.vocab_size = len(self.corp.dictionary.idx2word)
    self.EOS_ID = self.corp.dictionary.word2idx[self.corp.dictionary.EOS]
    self.end_token = self.corp.dictionary.word2idx[self.corp.dictionary.EOS]