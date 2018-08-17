# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import numpy as np
import os
import errno
import io
import shutil


def create_source(size, source_vocab, vocab_map):
  source = []
  for i in range(0, size):
    new_rol = []
    for j in range(0, np.random.randint(low=5, high=51)):
      new_dig = np.random.randint(low=0, high=len(vocab_map))
      new_rol.append(vocab_map[new_dig])
      if new_dig not in source_vocab:
        source_vocab[new_dig] = 0
      else:
        source_vocab[new_dig] += 1
    source.append(new_rol)
  return source


def create_target(size, source):
  target = []
  for i in range(0, size):
    new_row = list(reversed(source[i]))
    target.append(new_row)
  return target


def write_to_file(path, data):
  with io.open(path, 'w', encoding='utf-8') as f:
    for row in data:
      f.write(' '.join(row) + '\n')
  f.close()


def write_vocab_to_file(path, data, vocab_map):
  with io.open(path, 'w', encoding='utf-8') as f:
    for key, value in data.items():
      f.write(vocab_map[key]+'\t'+str(value)+'\n')
  f.close()


def create_directory(path):
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise


def create_data(train_corpus_size=10000, dev_corpus_size=1000,
                test_corpus_size=2000, data_path="./toy_text_data"):

  train_path = os.path.join(data_path, "train")
  dev_path = os.path.join(data_path, "dev")
  test_path = os.path.join(data_path, "test")
  vocab_path = os.path.join(data_path, "vocab")

  train_source_path = os.path.join(train_path, "source.txt")
  train_target_path = os.path.join(train_path, "target.txt")

  dev_source_path = os.path.join(dev_path, "source.txt")
  dev_target_path = os.path.join(dev_path, "target.txt")

  test_source_path = os.path.join(test_path, "source.txt")
  test_target_path = os.path.join(test_path, "target.txt")

  vocab_source_path = os.path.join(vocab_path, "source.txt")
  vocab_target_path = os.path.join(vocab_path, "target.txt")

  source_vocab = {}

  vocab_map = {0: '\u03B1',
               1: '\u03B2',
               2: '\u03B3',
               3: '\u03B4',
               4: '\u03B5',
               5: '\u03B6',
               6: '\u03B7',
               7: '\u03B8',
               8: '\u03B9',
               9: '\u03BA'}

  create_directory(train_path)
  create_directory(test_path)
  create_directory(dev_path)
  create_directory(vocab_path)

  train_source = create_source(train_corpus_size, source_vocab, vocab_map)
  write_to_file(train_source_path, train_source)
  write_to_file(
    train_target_path,
    create_target(train_corpus_size, train_source),
  )

  dev_source = create_source(dev_corpus_size, source_vocab, vocab_map)
  write_to_file(dev_source_path, dev_source)
  write_to_file(dev_target_path, create_target(dev_corpus_size, dev_source))

  test_source = create_source(test_corpus_size, source_vocab, vocab_map)
  write_to_file(test_source_path, test_source)
  write_to_file(test_target_path, create_target(test_corpus_size, test_source))

  write_vocab_to_file(vocab_source_path, source_vocab, vocab_map)
  # in our case, source and target vocabs are the same
  write_vocab_to_file(vocab_target_path, source_vocab, vocab_map)


def remove_data(data_path="./toy_text_data"):
  shutil.rmtree(data_path)


if __name__ == '__main__':
  create_data(data_path='./toy_text_data')
