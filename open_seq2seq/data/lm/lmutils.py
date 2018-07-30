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

    def __ledn__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, vocab_path, content_path=None):
        self.dictionary = Dictionary()
        vocab_file = open(vocab_path, 'r')
        n = int(vocab_file.readline().strip())
        self.dictionary.idx2word = [0 for _ in range(n)]
        for line in vocab_file:
            parts = line.strip().split('\t')
            token_id, word, count = int(parts[0]), parts[1], int(parts[2]) 
            self.dictionary.word2idx[word] = token_id
            self.dictionary.idx2word[token_id] = word
            self.dictionary.counter[token_id] = count

        self.content = None
        if content_path:
            self.content = self.load_ids(content_path)

    def load_ids(self, filename):
        ids = open(filename, 'r').read().strip().split('\t')
        return np.asarray([int(i) for i in ids])

    def tokenize(self, text):
        unk_id = self.dictionary.word2idx[self.dictionary.UNK]
        ids = []
        words = text.split()
        for word in words:
            ids.append(self.dictionary.word2idx.get(word, unk_id))

        return np.asarray(ids)
    
    def text_ids_to_string(self, ids):
        pass