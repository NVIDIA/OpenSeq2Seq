# Copyright (c) 2017 NVIDIA Corporation
"""Data Layer Classes"""
import abc
import six
import numpy as np
import random
import csv
import copy
import io

@six.add_metaclass(abc.ABCMeta)
class DataLayer:
  UNK_ID = 0  # out-of-vocabulary tokens will map there
  S_ID = 1  # special start of sentence token
  EOS_ID = 2  # special end of sentence token
  PAD_ID = 3  # special padding token
  OUT_OF_BUCKET = 1234567890
  END_OF_CHOICE = -100
  """Abstract class that specifies data access operations
  """
  @abc.abstractmethod
  def __init__(self, params):
    """Initialize data layer
    :param params: Python dictionary with options,
    specifying mini-batch shapes, padding, etc.
    """
    self._params = params

  @abc.abstractmethod
  def iterate_one_epoch(self):
    """
    Goes through the data one time.
    :return: yields rectangular 2D numpy array with mini-batch data
    """
    pass

  @property
  def params(self):
    return self._params

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
  UNK_ID = 0 # out-of-vocabulary tokens will map there
  S_ID = 1 # special start of sentence token
  EOS_ID = 2 # special end of sentence token
  PAD_ID = 3 # special padding token
  OUT_OF_BUCKET = 1234567890

  bucket_sizes = [60, 120, 180, 240, 300, 360]

  def __init__(self, params):
    super(ParallelDataInRamInputLayer, self).__init__(params)
    self._batch_size = self.params['batch_size'] * self.params["num_gpus"] if "num_gpus" in self.params else self.params["batch_size"]
    self.source_file = self.params['source_file']
    self.target_file = self.params['target_file']

    self.src_vocab_file = self.params['src_vocab_file']
    self.tgt_vocab_file = self.params['tgt_vocab_file']

    self.bucket_src = self.params['bucket_src']
    self.bucket_tgt = self.params['bucket_tgt']

    self._istrain = ("mode" not in self._params or self._params["mode"] == "train")
    self._bucket_order = [] #used in inference
    self._shuffle = ('shuffle' in self._params and self._params['shuffle'])

    self.load_pre_src_tgt_vocabs(self.src_vocab_file, self.tgt_vocab_file)
    self.load_corpus()

    self.create_idx_seq_associations()
    self.bucketize()

  @property
  def target_seq2idx(self):
    return self.tgt_seq2idx

  @property
  def source_seq2idx(self):
    return self.src_seq2idx

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def source_corpus(self):
      return self.src_corpus

  @property
  def target_corpus(self):
      return self.tgt_corpus

  @property
  def source_idx2seq(self):
      return self.src_idx2seq

  @property
  def target_idx2seq(self):
      return  self.tgt_idx2seq

  @staticmethod
  def load_file_word(path, vocab):
    """
    Load file word-by-word
    :param path: path to data
    :param vocab: vocabluary
    :return: list of sentences with S_ID and EOS_ID added
    """
    sentences = []
    with io.open(path, newline = '', encoding = 'utf-8') as f:
      corpus_reader = csv.reader(f, delimiter = ' ')
      for line in corpus_reader:
          sentences.append([ParallelDataInRamInputLayer.S_ID] + list(
              map(lambda word: vocab[word] if word in vocab else ParallelDataInRamInputLayer.UNK_ID, line)) +
                           [ParallelDataInRamInputLayer.EOS_ID])
    return sentences

  def load_corpus(self):
    """
    Load both source and target corpus
    :return:
    """
    self.src_corpus = self.load_file_word(self.source_file, self.src_seq2idx)
    self.tgt_corpus = self.load_file_word(self.target_file, self.tgt_seq2idx)


  def load_pre_existing_vocabulary(self, path):
    """
    Loads pre-existing vocabulary into memory
    :param path: path to vocabulary
    :return: vocabulary
    """
    idx = ParallelDataInRamInputLayer.PAD_ID + 1
    vocab = {}
    with io.open(path, newline='', encoding = 'utf-8') as f:
      vocab_reader = csv.reader(f, delimiter='\t')
      for seq in vocab_reader:
        vocab[seq[0]] = idx
        idx += 1
    return vocab

  def load_pre_src_tgt_vocabs(self, src_path, tgt_path):
    self.src_seq2idx = self.load_pre_existing_vocabulary(src_path)
    self.tgt_seq2idx = self.load_pre_existing_vocabulary(tgt_path)

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
      return input_seq + [ParallelDataInRamInputLayer.PAD_ID]*(bucket_size - len(input_seq))

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
    while curr_bucket<len(bucket_sizes) and input_size > bucket_sizes[curr_bucket]:
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

    if len(self.src_corpus)!=len(self.tgt_corpus):
      raise ValueError("""Source and target files for NMT must contain equal
                       amounts of sentences. But they contain %s and %s correspondingly.
                       """ % (len(self.src_corpus), len(self.tgt_corpus)))

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

      if not bucket_id in self._bucket_id_to_src_example:
        self._bucket_id_to_src_example[bucket_id] = []
      self._bucket_id_to_src_example[bucket_id].append(x)

      if not bucket_id in self._bucket_id_to_tgt_example:
        self._bucket_id_to_tgt_example[bucket_id] = []
      self._bucket_id_to_tgt_example[bucket_id].append(y)

      if not self._shuffle:
        self._bucket_order.append(bucket_id)

    print("WARNING: skipped %d pairs with max source size of %d and max target size of %d" %
                                                (skipped_count, skipped_max_source, skipped_max_target))
    self._bucket_sizes = {}
    for bucket_id in self._bucket_id_to_src_example.keys():
      self._bucket_sizes[bucket_id] = len(self._bucket_id_to_src_example[bucket_id])
      if self._shuffle:
        c = list(zip(self._bucket_id_to_src_example[bucket_id],
                self._bucket_id_to_tgt_example[bucket_id]))
        random.shuffle(c)
        a, b = zip(*c)
        self._bucket_id_to_src_example[bucket_id] = np.asarray(a)
        self._bucket_id_to_tgt_example[bucket_id] = np.asarray(b)
      else:
        self._bucket_id_to_src_example[bucket_id] = np.asarray(self._bucket_id_to_src_example[bucket_id])
        self._bucket_id_to_tgt_example[bucket_id] = np.asarray(self._bucket_id_to_tgt_example[bucket_id])

  @staticmethod
  def _pad_to_bucket_size(inseq, bucket_size):
    if len(inseq) == bucket_size:
        return inseq
    else:
        return inseq + [ParallelDataInRamInputLayer.PAD_ID] * (bucket_size - len(inseq))

  def iterate_one_epoch(self):
    """
    Iterates through the data ones
    :return: yield mini-batched x and y
    """
    from .utils import weighted_choice
    start_inds = {}
    choices = copy.deepcopy(self._bucket_sizes)
    for bucket_id in choices.keys():
        start_inds[bucket_id] = 0

    if self._shuffle:
        bucket_id = weighted_choice(choices)
    else:
        ordering = list(reversed(self._bucket_order))
        bucket_id = ordering.pop()

    while bucket_id != self.END_OF_CHOICE:
      end_ind = min(start_inds[bucket_id] + self.batch_size, self._bucket_id_to_src_example[bucket_id].shape[0])
      x = self._bucket_id_to_src_example[bucket_id][start_inds[bucket_id]:end_ind]
      len_x = np.asarray(list(map(lambda row: len(row), x)))
      x = np.vstack(
        map(lambda row: np.asarray(self._pad_to_bucket_size(row, np.max(len_x))), x))
      if self._istrain:
        y = self._bucket_id_to_tgt_example[bucket_id][start_inds[bucket_id]:end_ind]
        len_y = np.asarray(list(map(lambda row: len(row), y)))
        y = np.vstack(
            map(lambda row: np.asarray(self._pad_to_bucket_size(row, np.max(len_y))), y))
      else:
        y = None
        len_y = np.asarray([])
      yielded_examples = end_ind - start_inds[bucket_id]
      start_inds[bucket_id] += yielded_examples

      bucket_id_to_yield = bucket_id
      choices[bucket_id] -= yielded_examples
      if self._shuffle:
        bucket_id = weighted_choice(choices)
      elif len(ordering) > 0:
        bucket_id = ordering.pop()
      else:
        bucket_id = self.END_OF_CHOICE

      if self._istrain and yielded_examples < self.batch_size:
        continue
      yield x, y, bucket_id_to_yield, len_x, len_y

  def iterate_n_epochs(self, num_epochs):
    for epoch_ind in range(num_epochs):
      for x, y, bucket_id_to_yield, len_x, len_y in self.iterate_one_epoch():
        yield x, y, bucket_id_to_yield, len_x, len_y
