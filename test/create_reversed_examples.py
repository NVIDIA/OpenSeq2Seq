# Copyright (c) 2017 NVIDIA Corporation
import numpy as np
import os, errno

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

def create_source(size):
	global source_vocab
	source = []
	for i in range(0,size):
		new_rol = []
		for j in range(0, np.random.randint(low=5, high=51)):
			new_dig = np.random.randint(low=0, high=10)
			new_rol.append(new_dig)
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
	with open(path, 'w') as f:
		for row in data:
			f.write(str(row)[1:-1].replace(',','') + '\n')
	f.close()

def write_vocab_to_file(path, data):
	with open(path, 'w') as f:
		for key, value in data.items():
			f.write(str(key)+'\t'+str(value)+'\n')
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
