from collections import Counter
import glob
import os
import pathlib
import random
import re
import shutil

from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd

class Dictionary(object):
  '''
  Adapted from salesforce's repo:
  https://github.com/salesforce/awd-lstm-lm/blob/master/data.py
  '''
  def __init__(self, limit=3, vocab_link=None): # do we need limit?
    self.word2idx = {}
    self.idx2word = []
    self.counter = Counter()
    self.UNK = '<unk>'
    self.EOS = '<eos>'
    if vocab_link and os.path.isfile(vocab_link):
      self.load_vocab(vocab_link)

  def add_word(self, word):
    if word not in self.word2idx:
      self.idx2word.append(word)
      self.word2idx[word] = len(self.idx2word) - 1
    token_id = self.word2idx[word]
    self.counter[token_id] += 1
    return self.word2idx[word]

  def load_vocab(self, vocab_link):
    vocab_file = open(vocab_link, 'r')
    lines = vocab_file.readlines()
    n = int(lines[-1].strip())
    self.idx2word = [0 for _ in range(n)]
    for line in lines[:-1]:
      parts = line.strip().split('\t')
      token_id, word, count = int(parts[0]), parts[1], int(parts[2]) 
      self.word2idx[word] = token_id
      self.idx2word[token_id] = word
      self.counter[token_id] = count
    if not self.UNK in self.word2idx:
      self.add_word(self.UNK)
    if not self.EOS in self.word2idx:
      self.add_word(self.EOS)


  def __len__(self):
    return len(self.idx2word)

def check_exist(proc_path):
  filenames = ['train.ids', 'valid.ids', 'test.ids']
  paths = [os.path.join(proc_path, name) for name in filenames]
  paths.append(proc_path)
  for name in paths:
    if not os.path.exists(name):
      return False
  return True

def list2str(list):
  return '\t'.join([str(num) for num in list])

def unzip(data):
  tmp = [list(t) for t in zip(*data)]
  return (tmp[0], tmp[1])

class Corpus(object):
  def __init__(self, raw_path, proc_path, change_contraction=True, limit=3):
    pathlib.Path(proc_path).mkdir(exist_ok=True)
    self.limit = limit
    self.dictionary = Dictionary(limit)
    self.vocab_link = 'vocab.txt'
    exists = check_exist(proc_path)
    self.change_contraction = change_contraction

    if not exists:
      print('Creating corpus from raw data ...')
      if raw_path and 'raw' in raw_path:
        self._change_names(raw_path)
      if not raw_path:
        raise ValueError("data_root [directory to the original data] must be specified")
      self.preprocess(raw_path, proc_path)
      self.create_dictionary(proc_path, os.path.join(proc_path, 'train.txt'))
      self.dictionary = Dictionary(limit)
      self.dictionary.load_vocab(os.path.join(proc_path, self.vocab_link))
      self.train = self.tokenize(proc_path, proc_path, 'train.txt')
      self.valid = self.tokenize(proc_path, proc_path, 'valid.txt')
      self.test = self.tokenize(proc_path, proc_path, 'test.txt')
    else:
      self.load_corpus(proc_path)

  def _change_names(self, raw_path):
    if os.path.isfile(os.path.join(raw_path, 'wiki.train.raw')):
      os.rename(os.path.join(raw_path, 'wiki.train.raw'), os.path.join(raw_path, 'train.txt'))
      os.rename(os.path.join(raw_path, 'wiki.valid.raw'), os.path.join(raw_path, 'valid.txt'))
      os.rename(os.path.join(raw_path, 'wiki.test.raw'), os.path.join(raw_path, 'test.txt'))

  def preprocess(self, raw_path, proc_path):
    for filename in ['train.txt', 'valid.txt', 'test.txt']:
      in_ = open(os.path.join(raw_path, filename), 'r')
      out = open(os.path.join(proc_path, filename), 'w')
      for line in in_:
        line = re.sub('@-@', '-', line)
        line = re.sub('-', ' - ', line)
        line = re.sub('etc .', 'etc.', line)
        if self.change_contraction:
          line = re.sub("n 't", " n't", line)
        tokens = []
        for token in line.split():
          tokens.append(token.strip())
        out.write(' '.join(tokens) + '\n')

  def create_dictionary(self, proc_path, filename):
    '''
    Add words to the dictionary only if it's in the train file
    '''
    self.dictionary.add_word(self.dictionary.UNK)
    with open(filename, 'r') as f:
      f.readline()
      for line in f:
        words = line.split() + [self.dictionary.EOS]
        for word in words:
          self.dictionary.add_word(word)

    with open(os.path.join(proc_path, self.vocab_link), 'w') as f:
      f.write('\t'.join(['0', self.dictionary.UNK, '0']) + '\n')
      idx = 1
      for token_id, count in self.dictionary.counter.most_common():
        if count < self.limit:
          f.write(str(idx) + '\n')
          return
        f.write('\t'.join([str(idx), 
              self.dictionary.idx2word[token_id], 
              str(count)]) + '\n')
        idx += 1
      
  def tokenize(self, raw_path, proc_path, filename):
    unk_id = self.dictionary.word2idx[self.dictionary.UNK]
    out = open(os.path.join(proc_path, filename[:-3] + 'ids'), 'w')
    with open(os.path.join(raw_path, filename), 'r') as f:
      ids = []
      for line in f:
        words = line.split() + [self.dictionary.EOS]
        for word in words:
          ids.append(self.dictionary.word2idx.get(word, unk_id))
    out.write(list2str(ids))
    out.close()

    return np.asarray(ids)

  def load_ids(self, filename):
    ids = open(filename, 'r').read().strip().split('\t')
    return np.asarray([int(i) for i in ids])

  def list2str(self, list):
    return '\t'.join([str(num) for num in list])

  def load_corpus(self, proc_path):
    print('Loading corpus from processed data ...')
    self.dictionary.load_vocab(os.path.join(proc_path, self.vocab_link))
    self.train = self.load_ids(os.path.join(proc_path, 'train.ids'))
    self.valid = self.load_ids(os.path.join(proc_path, 'valid.ids'))
    self.test = self.load_ids(os.path.join(proc_path, 'test.ids'))

class IMDBCorpus(object):
  def __init__(self, raw_path, proc_path, lm_vocab_link, binary=True, get_stats=False):
    exists = check_exist(proc_path)
    pathlib.Path(proc_path).mkdir(exist_ok=True)
    self.dictionary = Dictionary(vocab_link=lm_vocab_link)
    self.binary = binary
    self.raw_path = raw_path
    self.proc_path = proc_path
    self._get_stats = get_stats

    if not exists:
      print('Creating corpus from raw data ...')
      if not raw_path:
        raise ValueError("data_root [directory to the original data] must be specified")
      self.preprocess()
    else:
      self.load_corpus(proc_path)

  def check_oov(self, txt):
    txt = txt.lower()
    txt = re.sub('thats', "that's", txt)
    txt = re.sub('wouldnt', "wounldn't", txt)
    txt = re.sub('couldnt', "couldn't", txt)
    txt = re.sub('cant', "can't", txt)
    txt = re.sub('dont', "don't", txt)
    txt = re.sub("didnt", "didn't", txt)
    txt = re.sub("isnt", "isn't", txt)
    txt = re.sub("wasnt", "wasn't", txt)
    return word_tokenize(txt)

  def tokenize(self, txt):
    txt = re.sub('<br />', ' ', txt)
    txt = re.sub('', ' ', txt)
    txt = re.sub('', ' ', txt)
    txt = re.sub('-', ' - ', txt)
    txt = re.sub('\.', ' . ', txt)
    txt = re.sub('\+', ' + ', txt)
    txt = re.sub('\*', ' * ', txt)
    txt = re.sub('/', ' / ', txt)
    txt = re.sub('`', "'", txt)
    txt = re.sub(' ms \.', " ms.", txt)
    txt = re.sub('Ms \.', "Ms.", txt)
    
    words = []
    for token in word_tokenize(txt):
      if not token in self.dictionary.word2idx:
        if token.startswith("'"):
          words.append("'")
          token = token[1:]
        if not token in self.dictionary.word2idx:
          tokens = self.check_oov(token)
          words.extend(tokens)
        else:
          words.append(token)
      else:
        words.append(token) 
    
    txt = ' '.join(words)
    txt = re.sub("''", '"', txt)
    txt = re.sub("' '", '"', txt)
    txt = re.sub("``", '"', txt)
    txt = re.sub('etc \.', 'etc. ', txt)
    txt = re.sub(' etc ', ' etc. ', txt)
    return txt

  def tokenize_folder(self, mode, token_file, rating_file):
    review_outfile = open(token_file, 'w')
    rating_outfile = open(rating_file, 'w')
    for sent in ['pos', 'neg']:
      files = glob.glob(os.path.join(self.raw_path, mode, sent, '*.txt'))
      for file in files:
        in_file = open(file, 'r')
        txt = self.tokenize(in_file.read())
        review_outfile.write(txt + "\n")
        if self.binary:
          if sent == 'pos':
            rating = "1"
          else:
            rating = "0"
        else:
          idx = file.rfind("_")
          rating = str(int(file[idx + 1:-4]) - 1)
        rating_outfile.write(rating + '\n')
        in_file.close()

  def txt2ids(self, mode, token_file, rating_file):
    if self._get_stats:
      import matplotlib
      matplotlib.use("TkAgg")
      from matplotlib import pyplot as plt
    rating_lines = open(rating_file, 'r').readlines()
    ratings = [int(line.strip()) for line in rating_lines]
    reviews = []
    unk_id = self.dictionary.word2idx[self.dictionary.UNK]
    unseen = []
    all_tokens = 0
    all_unseen = 0
    for line in open(token_file, 'r'):
      tokens = line.strip().split()
      reviews.append([self.dictionary.word2idx.get(token, unk_id) for token in tokens])
      if self._get_stats:
        for token in tokens:
          all_tokens += 1
          if not token in self.dictionary.word2idx:
            unseen.append(token)
            all_unseen += 1

    if self._get_stats:
      counter = Counter(unseen)

      out = open(os.path.join(self.proc_path, mode + '_unseen.txt'), 'w')
      for key, count in counter.most_common():
          out.write(key + '\t' + str(count) + '\n')

      lengths = np.asarray([len(review) for review in reviews])
      stat_file = open(os.path.join(self.proc_path, 'statistics.txt'), 'w')
      stat_file.write(mode + '\n')
      short_lengths = [l for l in lengths if l <= 256]
      stat_file.write('\t'.join(['Min', 'Max', 'Mean', 'Median', 'STD', 'Total', '<=256']) + '\n')
      stats = [np.min(lengths), np.max(lengths), np.mean(lengths), np.median(lengths), np.std(lengths), len(lengths), len(short_lengths)]
      stat_file.write('\t'.join([str(t) for t in stats]) + '\n')
      stat_file.write('Total {} unseen out of {} all tokens. Probability {}.\n'.
        format(all_unseen, all_tokens, all_unseen / all_tokens))
      plt.hist(lengths, bins=20)
      plt.savefig(os.path.join(self.proc_path, mode + '_hist.png'))
      plt.hist(short_lengths, bins=20)
      plt.savefig(os.path.join(self.proc_path, mode + '_short_hist.png'))

    return list(zip(reviews, ratings))

  def preprocess_folder(self, mode):
    token_file = os.path.join(self.proc_path, mode + '.tok')
    rating_file = os.path.join(self.proc_path, mode + '.inter.rat')
    self.tokenize_folder(mode, token_file, rating_file)
    return self.txt2ids(mode, token_file, rating_file)

  def partition(self, data, val_count=1000):
    random.shuffle(data)
    return data[val_count:], data[:val_count]

  def ids2file(self):
    for mode in ['train', 'valid', 'test']:
      data = getattr(self, mode)
      review_out = open(os.path.join(self.proc_path, mode + '.ids'), 'w')
      rating_out = open(os.path.join(self.proc_path, mode + '.rat'), 'w')
      for review, rating in data:
        review_out.write(list2str(review) + '\n')
        rating_out.write(str(rating) + '\n')

  def preprocess(self):
    os.makedirs(self.proc_path, exist_ok=True)
    train = self.preprocess_folder('train')
    self.train, self.valid = self.partition(train)
    self.test = self.preprocess_folder('test')
    self.ids2file()

  def load_ids(self, mode):
    review_lines = open(os.path.join(self.proc_path, mode + '.ids')).readlines()
    rating_lines = open(os.path.join(self.proc_path, mode + '.rat')).readlines()
    ratings = [int(line.strip()) for line in rating_lines]
    reviews = [[int(i) for i in line.strip().split('\t')] for line in review_lines]
    return list(zip(reviews, ratings))

  def load_corpus(self, proc_path):
    print('Loading corpus from processed data ...')
    self.train = self.load_ids('train')
    self.valid = self.load_ids('valid')
    self.test = self.load_ids('test')

class SSTCorpus(object):
  def __init__(self, raw_path, proc_path, lm_vocab_link, get_stats=False):
    exists = check_exist(proc_path)
    pathlib.Path(proc_path).mkdir(exist_ok=True)
    self.dictionary = Dictionary(vocab_link=lm_vocab_link)
    self.raw_path = raw_path
    self.proc_path = proc_path
    self._get_stats = get_stats

    if not exists:
      print('Creating corpus from raw data ...')
      if not raw_path:
        raise ValueError("data_root [directory to the original data] must be specified")
      self.preprocess()
    else:
      self.load_corpus(proc_path)

  def check_oov(self, txt):
    txt = txt.lower()
    txt = re.sub('thats', "that's", txt)
    txt = re.sub('wouldnt', "wounldn't", txt)
    txt = re.sub('couldnt', "couldn't", txt)
    txt = re.sub('cant', "can't", txt)
    txt = re.sub('dont', "don't", txt)
    txt = re.sub("didnt", "didn't", txt)
    txt = re.sub("isnt", "isn't", txt)
    txt = re.sub("wasnt", "wasn't", txt)
    return word_tokenize(txt)

  def tokenize(self, txt):
    txt = re.sub('-', ' - ', txt)
    txt = re.sub('\+', ' + ', txt)
    txt = re.sub('\*', ' * ', txt)
    txt = re.sub('/', ' / ', txt)
    txt = re.sub('`', "'", txt)
    
    words = []
    for token in word_tokenize(txt):
      if not token in self.dictionary.word2idx:
        if token.startswith("'"):
          words.append("'")
          token = token[1:]
        if not token in self.dictionary.word2idx:
          tokens = self.check_oov(token)
          words.extend(tokens)
        else:
          words.append(token)
      else:
        words.append(token) 
    
    txt = ' '.join(words)
    txt = re.sub("''", '"', txt)
    txt = re.sub("' '", '"', txt)
    txt = re.sub("``", '"', txt)
    txt = re.sub('etc \.', 'etc. ', txt)
    txt = re.sub(' etc ', ' etc. ', txt)
    return txt

  def tokenize_file(self, mode):
    data = pd.read_csv(os.path.join(self.raw_path, mode + '.csv'))

    if mode == 'val':
      mode = 'valid'
    review_file = open(os.path.join(self.proc_path, mode + '.tok'), 'w')
    rating_file = open(os.path.join(self.proc_path, mode + '.rat'), 'w')
    for _, row in data.iterrows():
      review = self.tokenize(row['sentence'])
      review_file.write(review + '\n')
      rating_file.write(str(row['label']) + '\n')

  def txt2ids(self, mode):
    if self._get_stats:
      import matplotlib
      matplotlib.use("TkAgg")
      from matplotlib import pyplot as plt

    reviews = []
    unk_id = self.dictionary.word2idx[self.dictionary.UNK]
    unseen = []
    all_tokens = 0
    all_unseen = 0

    rating_lines = open(os.path.join(self.proc_path, mode + '.rat'), 'r').readlines()
    ratings = [int(line.strip()) for line in rating_lines]

    for line in open(os.path.join(self.proc_path, mode + '.tok'), 'r'):
      tokens = line.strip().split()
      reviews.append([self.dictionary.word2idx.get(token, unk_id) for token in tokens])
      if self._get_stats:
        for token in tokens:
          all_tokens += 1
          if not token in self.dictionary.word2idx:
            unseen.append(token)
            all_unseen += 1

    if self._get_stats:
      counter = Counter(unseen)

      out = open(os.path.join(self.proc_path, mode + '_unseen.txt'), 'w')
      for key, count in counter.most_common():
          out.write(key + '\t' + str(count) + '\n')

      lengths = np.asarray([len(review) for review in reviews])
      stat_file = open(os.path.join(self.proc_path, 'statistics.txt'), 'a')
      stat_file.write(mode + '\n')
      short_lengths = [l for l in lengths if l <= 96]
      stat_file.write('\t'.join(['Min', 'Max', 'Mean', 'Median', 'STD', 'Total', '<=96']) + '\n')
      stats = [np.min(lengths), np.max(lengths), np.mean(lengths), np.median(lengths), np.std(lengths), len(lengths), len(short_lengths)]
      stat_file.write('\t'.join([str(t) for t in stats]) + '\n')
      stat_file.write('Total {} unseen out of {} all tokens. Probability {}.\n'.
        format(all_unseen, all_tokens, all_unseen / all_tokens))
      plt.hist(lengths, bins=20)
      plt.savefig(os.path.join(self.proc_path, mode + '_hist.png'))
      plt.hist(short_lengths, bins=20)
      plt.savefig(os.path.join(self.proc_path, mode + '_short_hist.png'))

    return list(zip(reviews, ratings))

  def preprocess_file(self, mode):
    self.tokenize_file(mode)
    if mode == 'val':
      mode = 'valid'
    return self.txt2ids(mode)

  def ids2file(self):
    for mode in ['train', 'valid', 'test']:
      data = getattr(self, mode)
      review_out = open(os.path.join(self.proc_path, mode + '.ids'), 'w')
      rating_out = open(os.path.join(self.proc_path, mode + '.rat'), 'w')
      for review, rating in data:
        review_out.write(list2str(review) + '\n')
        rating_out.write(str(rating) + '\n')

  def preprocess(self):
    os.makedirs(self.proc_path, exist_ok=True)
    self.train = self.preprocess_file('train')
    self.valid = self.preprocess_file('val')
    self.test = self.preprocess_file('test')
    self.ids2file()

  def load_ids(self, mode):
    review_lines = open(os.path.join(self.proc_path, mode + '.ids')).readlines()
    rating_lines = open(os.path.join(self.proc_path, mode + '.rat')).readlines()
    ratings = [int(line.strip()) for line in rating_lines]
    reviews = [[int(i) for i in line.strip().split('\t')] for line in review_lines]
    return list(zip(reviews, ratings))

  def load_corpus(self, proc_path):
    print('Loading corpus from processed data ...')
    self.train = self.load_ids('train')
    self.valid = self.load_ids('valid')
    self.test = self.load_ids('test')

# SSTCorpus('/home/chipn/data/binary_sst', 'sst-processed-data-wkt2' , '/home/chipn/dev/OpenSeq2Seq/wkt2-processed-data/vocab.txt')
# SSTCorpus('/home/chipn/data/binary_sst', 'sst-processed-data-wkt103' , '/home/chipn/dev/OpenSeq2Seq/wkt103-processed-data/vocab.txt')
# IMDBCorpus('/home/chipn/data/aclImdb', 'imdb-processed-data-wkt103' , '/home/chipn/dev/OpenSeq2Seq/wkt103-processed-data/vocab.txt')
# IMDBCorpus('/home/chipn/data/aclImdb', 'imdb-processed-data-wkt2' , '/home/chipn/dev/OpenSeq2Seq/wkt2-processed-data/vocab.txt')