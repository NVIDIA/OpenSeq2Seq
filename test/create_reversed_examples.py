# Copyright (c) 2017 NVIDIA Corporation
from __future__ import unicode_literals
import numpy as np
import os, errno
import io

TRAIN_CORPUS_SIZE = 10000
DEV_CORPUS_SIZE = 1000
TEST_CORPUS_SIZE = 2000

DATA_PATH = "test/toy_data/"
TRAIN_PATH = "test/toy_data/train/"
TEST_PATH = "test/toy_data/test/"
DEV_PATH = "test/toy_data/dev/"
VOCAB_PATH = "test/toy_data/vocab/"

TRAIN_SOURCE_PATH = TRAIN_PATH + "source.txt"
TRAIN_TARGET_PATH = TRAIN_PATH + "target.txt"

DEV_SOURCE_PATH = DEV_PATH + "source.txt"
DEV_TARGET_PATH = DEV_PATH + "target.txt"

TEST_SOURCE_PATH = TEST_PATH + "source.txt"
TEST_TARGET_PATH = TEST_PATH + "target.txt"

VOCAB_SOURCE_PATH = VOCAB_PATH + "source.txt"
VOCAB_TARGET_PATH = VOCAB_PATH + "target.txt"

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

def create_source(size):
	global source_vocab
	source = []
	for i in range(0,size):
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
	for i in range(0,size):
		new_row = list(reversed(source[i]))
		target.append(new_row)
	return target

def write_to_file(path, data):
	with io.open(path, 'w', encoding='utf-8') as f:
		for row in data:
			f.write(' '.join(row) + '\n')
	f.close()

def write_vocab_to_file(path, data):
	with io.open(path, 'w', encoding='utf-8') as f:
		for key, value in data.items():
			f.write(vocab_map[key]+'\t'+str(value)+'\n')
	f.close()

def create_directory(path):
	directory = os.path.dirname(path)
	try:
	    os.makedirs(directory)
	except OSError as e:
	    if e.errno != errno.EEXIST:
	        raise

create_directory(DATA_PATH)
create_directory(TRAIN_PATH)
create_directory(TEST_PATH)
create_directory(DEV_PATH)
create_directory(VOCAB_PATH)

train_source = create_source(TRAIN_CORPUS_SIZE)
write_to_file(TRAIN_SOURCE_PATH, train_source)
write_to_file(TRAIN_TARGET_PATH, create_target(TRAIN_CORPUS_SIZE, train_source))

dev_source = create_source(DEV_CORPUS_SIZE)
write_to_file(DEV_SOURCE_PATH, dev_source)
write_to_file(DEV_TARGET_PATH, create_target(DEV_CORPUS_SIZE, dev_source))

test_source = create_source(TEST_CORPUS_SIZE)
write_to_file(TEST_SOURCE_PATH, test_source)
write_to_file(TEST_TARGET_PATH, create_target(TEST_CORPUS_SIZE, test_source))

write_vocab_to_file(VOCAB_SOURCE_PATH, source_vocab)
# in our case, source and target vocabs are the same
write_vocab_to_file(VOCAB_TARGET_PATH, source_vocab)
