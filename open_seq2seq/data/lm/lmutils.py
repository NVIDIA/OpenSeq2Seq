from collections import Counter
import os
import pathlib

import numpy as np

class Dictionary(object):
    '''
    Adapted from salesforce's repo:
    https://github.com/salesforce/awd-lstm-lm/blob/master/data.py
    '''
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.UNK = '<unk>'
        self.EOS = '<eos>'

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, raw_path, proc_path):
        pathlib.Path(proc_path).mkdir(exist_ok=True)
        self.dictionary = Dictionary()
        self.vocab_link = 'vocab.txt'
        exists = self.check_exist(proc_path)

        if not exists:
            print('Creating corpus from raw data ...')
            if not raw_path:
                raise ValueError("data_root [directory to the original data] must be specified")
            self.create_dictionary(proc_path, os.path.join(raw_path, 'train.txt'))
            self.train = self.tokenize(raw_path, proc_path, 'train.txt')
            self.valid = self.tokenize(raw_path, proc_path, 'valid.txt')
            self.test = self.tokenize(raw_path, proc_path, 'test.txt')
        else:
            self.load_corpus(proc_path)

    def check_exist(self, proc_path):
        paths = [proc_path, proc_path + '/vocab.txt', proc_path + '/train.ids', 
                proc_path + '/valid.ids', proc_path + '/test.ids']
        for name in paths:
            if not os.path.exists(name):
                return False
        return True

    def create_dictionary(self, proc_path, filename):
        '''
        Add words to the dictionary only if it's train file
        '''
        with open(filename, 'r') as f:
            f.readline()
            for line in f:
                words = line.split() + [self.dictionary.EOS]
                for word in words:
                    self.dictionary.add_word(word)

        with open(os.path.join(proc_path, self.vocab_link), 'w') as f:
            f.write(str(len(self.dictionary)) + '\n')
            for token_id, count in self.dictionary.counter.most_common():
                f.write('\t'.join([str(token_id), 
                            self.dictionary.idx2word[token_id], 
                            str(count)]) + '\n')


    def tokenize(self, raw_path, proc_path, filename):
        unk_id = self.dictionary.word2idx[self.dictionary.UNK]
        out = open(os.path.join(proc_path, filename[:-3] + 'ids'), 'w')
        with open(os.path.join(raw_path, filename), 'r') as f:
            ids = []
            for line in f:
                words = line.split() + [self.dictionary.EOS]
                for word in words:
                    ids.append(self.dictionary.word2idx.get(word, unk_id))
        out.write(self.list2str(ids)) #TODO: change to pickle
        out.close()

        return np.asarray(ids)

    def load_ids(self, filename):
        ids = open(filename, 'r').read().strip().split('\t')
        return np.asarray([int(i) for i in ids])

    def list2str(self, list):
        return '\t'.join([str(num) for num in list])

    def load_corpus(self, proc_path):
        print('Loading corpus from processed data ...')
        vocab_file = open(os.path.join(proc_path, self.vocab_link), 'r')
        n = int(vocab_file.readline().strip())
        self.dictionary.idx2word = [0 for _ in range(n)]
        for line in vocab_file:
            parts = line.strip().split('\t')
            token_id, word, count = int(parts[0]), parts[1], int(parts[2]) 
            self.dictionary.word2idx[word] = token_id
            self.dictionary.idx2word[token_id] = word
            self.dictionary.counter[token_id] = count
        self.train = self.load_ids(os.path.join(proc_path, 'train.ids'))
        self.valid = self.load_ids(os.path.join(proc_path, 'valid.ids'))
        self.test = self.load_ids(os.path.join(proc_path, 'test.ids'))