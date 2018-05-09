# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import numpy as np
import tensorflow as tf
import random
import copy
import io
from enum import Enum
from .data_layer import DataLayer
from .utils import load_pre_existing_vocabulary, pad_vocab_to_eight


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


class ParallelDataInRamInputLayer(DataLayer):
  """Parallel data layer class. It should be provided with:
  a) source vocabulary file b) target vocabulary file
  c) tokenized source file d) tokenized target file
  This class performs:
    1) loading of data, mapping tokens to their ids.
    2) Inserting special tokens, if needed
    3) Padding
    4) Bucketing
    5) Mini-batch construction
  All parameters for above actions should come through "params" dictionary
  passed to constructor.
  This class loads all data and serves it from RAM
  """
  UNK_ID = SpecialTextTokens.UNK_ID.value  # out-of-vocab tokens will map there
  S_ID = SpecialTextTokens.S_ID.value  # special start of sentence token
  EOS_ID = SpecialTextTokens.EOS_ID.value  # special end of sentence token
  PAD_ID = SpecialTextTokens.PAD_ID.value  # special padding token
  OUT_OF_BUCKET = SpecialTextTokens.OUT_OF_BUCKET.value
  END_OF_CHOICE = SpecialTextTokens.END_OF_CHOICE.value
  bucket_sizes = [60, 120, 180, 240, 300, 360]

  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'source_file': str,
      'target_file': str,
      'src_vocab_file': str,
      'tgt_vocab_file': str,
      'bucket_src': list,
      'bucket_tgt': list,
      'delimiter': str,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'time_major': bool,
      'pad_vocab_to_eight': bool,
    })

  def __init__(self, params, model, num_workers=None, worker_id=None):
    super(ParallelDataInRamInputLayer, self).__init__(params, model,
                                                      num_workers, worker_id)
    self._batch_size = self.params['batch_size']
    self._num_workers = num_workers
    self._worker_id = worker_id

    self.source_file = self.params['source_file']
    self.target_file = self.params['target_file']

    self.src_vocab_file = self.params['src_vocab_file']
    self.tgt_vocab_file = self.params['tgt_vocab_file']

    self.bucket_src = self.params['bucket_src']
    self.bucket_tgt = self.params['bucket_tgt']

    self._use_targets = self.params['mode'] != 'infer'
    self._bucket_order = []  # used in inference

    self.load_pre_src_tgt_vocabs(self.src_vocab_file, self.tgt_vocab_file)
    self.load_corpus()
    self.time_major = self.params.get("time_major", False)
    self.create_idx_seq_associations()

    self.params['src_vocab_size'] = len(self.src_seq2idx)
    self.params['tgt_vocab_size'] = len(self.tgt_seq2idx)

    self.params['target_seq2idx'] = self.tgt_seq2idx
    self.params['source_seq2idx'] = self.src_seq2idx
    self.params['target_idx2seq'] = self.tgt_idx2seq
    self.params['source_idx2seq'] = self.src_idx2seq
    self.params['target_corpus'] = self.tgt_corpus
    self.params['source_corpus'] = self.src_corpus

    self.iterator = None

  def build_graph(self):
    self.bucketize()
    self.iterator = self._iterate_one_epoch()

  def gen_input_tensors(self):
    # placeholders for feeding data
    len_shape = [self._batch_size]
    if self.time_major:
      seq_shape = [None, self._batch_size]
    else:
      seq_shape = [self._batch_size, None]

    x = tf.placeholder(tf.int32, seq_shape)
    x_length = tf.placeholder(tf.int32, len_shape)
    if self._use_targets:
      y = tf.placeholder(tf.int32, seq_shape)
      y_length = tf.placeholder(tf.int32, len_shape)
      return [x, x_length, y, y_length]
    else:
      return [x, x_length]

  def get_size_in_samples(self):
    return len(self.src_corpus)

  def load_file_word(self, path, vocab):
    """
    Load file word-by-word
    :param path: path to data
    :param vocab: vocabulary
    :return: list of sentences with S_ID and EOS_ID added
    """
    sentences = []
    with io.open(path, newline='', encoding='utf-8') as f:
      ind = 0
      for raw_line in f:
        line = raw_line.rstrip().split(' ')
        if self._num_workers is not None:
          # we will load 1/self._num_workers of the epoch
          if ind % self._num_workers != self._worker_id:
            ind += 1
            continue
        unk_id = ParallelDataInRamInputLayer.UNK_ID
        sentences.append(
          [ParallelDataInRamInputLayer.S_ID] + list(
            map(
              lambda word: vocab[word] if word in vocab else unk_id,
              line,
            )
          ) + [ParallelDataInRamInputLayer.EOS_ID]
        )
        ind += 1
    if self._num_workers is not None:
      print('******** Data layer on rank {} loaded {} sentences'.format(
        self._worker_id, len(sentences),
      ))
    return sentences

  def load_corpus(self):
    """
    Load both source and target corpus
    :return:
    """
    self.src_corpus = self.load_file_word(self.source_file, self.src_seq2idx)
    self.tgt_corpus = self.load_file_word(self.target_file, self.tgt_seq2idx)

  def load_pre_src_tgt_vocabs(self, src_path, tgt_path):
    self.src_seq2idx = load_pre_existing_vocabulary(src_path,
                                                    min_idx=ParallelDataInRamInputLayer.PAD_ID+1)
    self.tgt_seq2idx = load_pre_existing_vocabulary(tgt_path,
                                                    min_idx=ParallelDataInRamInputLayer.PAD_ID + 1)

  def create_idx_seq_associations(self):
    """
    Creates id to seq mappings
    :return:
    """
    # unknown
    self.src_seq2idx['<UNK>'] = ParallelDataInRamInputLayer.UNK_ID
    self.tgt_seq2idx['<UNK>'] = ParallelDataInRamInputLayer.UNK_ID
    # sentence start
    self.src_seq2idx['<S>'] = ParallelDataInRamInputLayer.S_ID
    self.tgt_seq2idx['<S>'] = ParallelDataInRamInputLayer.S_ID
    # sentence end
    self.src_seq2idx['</S>'] = ParallelDataInRamInputLayer.EOS_ID
    self.tgt_seq2idx['</S>'] = ParallelDataInRamInputLayer.EOS_ID
    # padding
    self.src_seq2idx['<PAD>'] = ParallelDataInRamInputLayer.PAD_ID
    self.tgt_seq2idx['<PAD>'] = ParallelDataInRamInputLayer.PAD_ID

    if self.params.get('pad_vocab_to_eight', False):
      self.src_seq2idx = pad_vocab_to_eight(self.src_seq2idx)
      self.tgt_seq2idx = pad_vocab_to_eight(self.tgt_seq2idx)

    self.src_idx2seq = {id: w for w, id in self.src_seq2idx.items()}
    self.tgt_idx2seq = {id: w for w, id in self.tgt_seq2idx.items()}

  @staticmethod
  def pad_sequences(input_seq, bucket_size):
    """
    Pads sequence
    :param input_seq: sequenc to pad
    :param bucket_size: size of the bucket
    :return: padded sequence
    """
    if len(input_seq) == bucket_size:
      return input_seq
    else:
      return input_seq + [ParallelDataInRamInputLayer.PAD_ID] * \
             (bucket_size - len(input_seq))

  def determine_bucket(self, input_size, bucket_sizes):
    """
    Given input size and bucket sizes, determines closes bucket
    :param input_size: size of the input sequence
    :param bucket_sizes: list of bucket sizes
    :return: best bucket id
    """
    if len(bucket_sizes) <= 0:
      raise ValueError("No buckets specified")
    curr_bucket = 0
    while curr_bucket < len(bucket_sizes) and \
            input_size > bucket_sizes[curr_bucket]:
      curr_bucket += 1
    if curr_bucket >= len(bucket_sizes):
      return ParallelDataInRamInputLayer.OUT_OF_BUCKET
    else:
      return curr_bucket

  def bucketize(self):
    """
    Put data into buckets
    :return:
    """
    self._bucket_id_to_src_example = {}
    self._bucket_id_to_tgt_example = {}
    skipped_count = 0
    skipped_max_source = 0
    skipped_max_target = 0

    if len(self.src_corpus) != len(self.tgt_corpus):
      raise ValueError(
        """Source and target files for NMT must contain equal
        amounts of sentences. But they contain %s and %s correspondingly.
        """ % (len(self.src_corpus), len(self.tgt_corpus))
      )

    for src, tgt in zip(self.src_corpus, self.tgt_corpus):
      bucket_id = max(self.determine_bucket(len(src), self.bucket_src),
                      self.determine_bucket(len(tgt), self.bucket_tgt))

      if bucket_id == ParallelDataInRamInputLayer.OUT_OF_BUCKET:
        skipped_count += 1
        skipped_max_source = np.maximum(skipped_max_source, len(src))
        skipped_max_target = np.maximum(skipped_max_target, len(tgt))
        continue
      x = src
      y = tgt

      if bucket_id not in self._bucket_id_to_src_example:
        self._bucket_id_to_src_example[bucket_id] = []
      self._bucket_id_to_src_example[bucket_id].append(x)

      if bucket_id not in self._bucket_id_to_tgt_example:
        self._bucket_id_to_tgt_example[bucket_id] = []
      self._bucket_id_to_tgt_example[bucket_id].append(y)

      if not self.params['shuffle']:
        self._bucket_order.append(bucket_id)

    print(
      "WARNING: skipped %d pairs with max source size" 
      "of %d and max target size of %d" % (skipped_count,
                                           skipped_max_source,
                                           skipped_max_target)
    )
    self._bucket_sizes = {}
    for bucket_id in self._bucket_id_to_src_example.keys():
      self._bucket_sizes[bucket_id] = len(
        self._bucket_id_to_src_example[bucket_id]
      )
      if self.params['shuffle']:
        c = list(zip(self._bucket_id_to_src_example[bucket_id],
                 self._bucket_id_to_tgt_example[bucket_id]))
        random.shuffle(c)
        a, b = zip(*c)
        self._bucket_id_to_src_example[bucket_id] = np.asarray(a)
        self._bucket_id_to_tgt_example[bucket_id] = np.asarray(b)
      else:
        self._bucket_id_to_src_example[bucket_id] = np.asarray(
          self._bucket_id_to_src_example[bucket_id]
        )
        self._bucket_id_to_tgt_example[bucket_id] = np.asarray(
          self._bucket_id_to_tgt_example[bucket_id]
        )

  @staticmethod
  def _pad_to_bucket_size(inseq, bucket_size):
    if len(inseq) == bucket_size:
        return inseq
    else:
        return inseq + [ParallelDataInRamInputLayer.PAD_ID] * \
               (bucket_size - len(inseq))

  def next_batch(self):
    try:
      return next(self.iterator)
    except StopIteration:
      self.iterator = self._iterate_one_epoch()
      return next(self.iterator)

  def next_batch_feed_dict(self):
    return {self.get_input_tensors(): self.next_batch()}

  def shuffle(self):
    # doing nothing since all happens inside _iterate_one_epoch
    pass

  def _iterate_one_epoch(self):
    """
    Iterates through the data ones
    :return: yield mini-batched x and y
    """
    start_inds = {}
    choices = copy.deepcopy(self._bucket_sizes)
    for bucket_id in choices.keys():
        start_inds[bucket_id] = 0

    if self.params['shuffle']:
        bucket_id = weighted_choice(choices)
    else:
        ordering = list(reversed(self._bucket_order))
        bucket_id = ordering.pop()

    while bucket_id != self.END_OF_CHOICE:
      end_ind = min(
        start_inds[bucket_id] + self._batch_size,
        self._bucket_id_to_src_example[bucket_id].shape[0],
      )
      x = self._bucket_id_to_src_example[bucket_id][start_inds[bucket_id]:end_ind]
      len_x = np.asarray(list(map(lambda row: len(row), x)))
      if start_inds[bucket_id] >= end_ind:
        bucket_id = (weighted_choice(choices))
        print('Eval dl corner case')
        continue
      x = np.vstack(map(
        lambda row: np.asarray(self._pad_to_bucket_size(row, np.max(len_x))),
        x,
      ))
      if self._use_targets:
        y = self._bucket_id_to_tgt_example[bucket_id][start_inds[bucket_id]:end_ind]
        len_y = np.asarray(list(map(lambda row: len(row), y)))
        y = np.vstack(map(
          lambda row: np.asarray(self._pad_to_bucket_size(row, np.max(len_y))),
          y,
        ))
      else:
        y = None
        len_y = np.asarray([])
      yielded_examples = end_ind - start_inds[bucket_id]
      start_inds[bucket_id] += yielded_examples

      choices[bucket_id] -= yielded_examples
      if self.params['shuffle']:
        bucket_id = weighted_choice(choices)
      elif len(ordering) > 0:
        bucket_id = ordering.pop()
      else:
        bucket_id = self.END_OF_CHOICE

      if self._use_targets and yielded_examples < self._batch_size:
        continue
      if self._use_targets:
        if self.time_major:
          x = x.transpose()
          y = y.transpose()
        yield x, len_x, y, len_y
      else:
        if self.time_major:
          x = x.transpose()
        yield x, len_x
    self.bucketize()


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
    self._use_targets = self.params['mode'] != 'infer'
    if not self._use_targets:
      self.target_file = self.source_file
      if 'target_file' in self.params:
        print("WARNING: target file was specified but was "
              "ignored by data layer because 'mode' == 'infer'")
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

    def read_file(fname):
      lines = []
      with open(fname, 'r') as fin:
        for line in fin:
          lines.append(line)
      return lines

    self.source_lines = self.split_data(read_file(self.source_file))
    self.target_lines = self.split_data(read_file(self.target_file))

    self.dataset_size = len(self.source_lines)

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

    _sources = tf.data.Dataset.from_tensor_slices(self.source_lines)\
      .map(lambda line: tf.py_func(func=src_token_to_id, inp=[line],
                                   Tout=[tf.int32], stateful=False),
           num_parallel_calls=self._map_parallel_calls) \
      .map(lambda tokens: (tokens, tf.size(tokens)),
           num_parallel_calls=self._map_parallel_calls)

    _targets = tf.data.Dataset.from_tensor_slices(self.target_lines) \
      .map(lambda line: tf.py_func(func=tgt_token_to_id, inp=[line],
                                   Tout=[tf.int32], stateful=False),
           num_parallel_calls=self._map_parallel_calls) \
      .map(lambda tokens: (tokens, tf.size(tokens)),
           num_parallel_calls=self._map_parallel_calls)

    _src_tgt_dataset = tf.data.Dataset.zip((_sources, _targets)).filter(
      lambda t1, t2: tf.logical_and(tf.less_equal(t1[1], self.max_len),
                                    tf.less_equal(t2[1], self.max_len))
    ).cache()

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
