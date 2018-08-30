lm_vocab = '/home/chipn/dev/OpenSeq2Seq/wkt2_processed_data/vocab.txt'
# lm_vocab = '/home/chipn/dev/nlp-master/wikitext-103/vocab.txt'
folder = '/home/chipn/data/aclImdb'
processed_fold = 'imdb_processed_data'

import glob
import os
import re

from nltk.tokenize import word_tokenize

train_fold = os.path.join(folder, 'train')
test_fold = os.path.join(folder, 'test')

def compare_vocab(lm_vocab, imdb_vocab):
    lines = open(lm_vocab, 'r').readlines()
    n = int(lines[-1].strip())
    
    lm_dict = {}
    for line in lines[:-1]:
        parts = line.strip().split('\t')
        token_id, word, count = int(parts[0]), parts[1], int(parts[2]) 
        lm_dict[word.lower()] = token_id

    imdb = open(imdb_vocab, 'r').readlines()
    imdb_dict = {imdb[i].strip(): i for i in range(len(imdb))}


    print(len(lm_dict), len(imdb_dict))
    print(len(lm_dict.keys() & imdb_dict.keys()))


# def tokenize(txt, normalize_digits=False):
    # txt = re.sub('<br />', ' ', txt)
    # words = []
    # _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    # _DIGIT_RE = re.compile(r"\d")
    # for fragment in txt.strip().split():
    #     for token in re.split(_WORD_SPLIT, fragment):
    #         if not token:
    #             continue
    #         if normalize_digits:
    #             token = re.sub(_DIGIT_RE, '#', token)
    #         words.append(token)
    # txt = ' '.join(words)
    # txt = re.sub("' ll", "'ll", txt)
    # return txt

def tokenize(txt, normalize_digits=False):
    txt = re.sub('<br />', ' ', txt)
    words = []
    for word in word_tokenize(txt):
       word = re.sub('-', ' - ', word)
       words.extend(word.split())
    txt = ' '.join(words)
    txt = re.sub("``", '"', txt)
    txt = re.sub("''", '"', txt)
    return txt

def get_files(fold, mode, sent):
    files = glob.glob(os.path.join(fold, mode, sent, '*.txt'))
    out_fold = os.path.join(processed_fold, mode, sent)
    os.makedirs(out_fold, exist_ok=True)
    for file in files[:10]:
        idx = file.rfind("/")
        filename = file[idx + 1:]
        in_file = open(file, 'r')
        out_file = open(os.path.join(out_fold, filename), 'w')
        txt = tokenize(in_file.read())
        out_file.write(txt)
        in_file.close()
        out_file.close()

def preprocess(fold, mode):
    neg_files = get_files(fold, mode, 'neg')

    
    

# compare_vocab(lm_vocab, os.path.join(folder, 'imdb.vocab'))
preprocess(folder, 'train')

# def test(txt):
#     a = re.compile(r"\w-\w")
#     re.sub(a, "\w @-@ \w', txt)