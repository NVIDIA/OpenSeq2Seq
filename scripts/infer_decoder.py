import argparse

import pickle
import numpy as np

from ctc_decoders import Scorer
from ctc_decoders import ctc_greedy_decoder
from ctc_decoders import ctc_beam_search_decoder_batch, ctc_beam_search_decoder
from collections import defaultdict
import pandas as pd

parser = argparse.ArgumentParser(
  description='CTC decoding and tuning with LM rescoring'
)
parser.add_argument('--mode',
                    help='either \'greedy\' (default) or \'beam search with lm\'',
                    default='greedy'
                    )
parser.add_argument('--infer_output_file',
                    help='output CSV file for \'infer\' mode',
                    required=True
                    )
parser.add_argument('--logits',
                    help='pickle file with CTC logits',
                    required=True
                    )
parser.add_argument('--labels',
                    help='CSV file with audio filenames \
      (and ground truth transcriptions for \'eval\' mode)',
                    required=True
                    )

parser.add_argument('--alpha', type=float,
                    help='value of LM weight',
                    required=True
                    )

parser.add_argument('--beta', type=float,
                    help='value of word count weight',
                    required=True
                    )

parser.add_argument('--beam_width', type=int,
                    help='beam width for beam search decoder',
                    required=False, default=128
                    )
args = parser.parse_args()


def load_dump(pickle_file):
  with open(pickle_file, 'rb') as f:
    data = pickle.load(f, encoding='bytes')
  logits = data["logits"]
  # step_size = data["step_size"]
  vocab = data["vocab"]
  return logits, vocab


def load_labels(csv_file):
  labels = pd.read_csv(csv_file)
  files = labels["File"]
  lms = labels["lm"]
  return files, lms


def load_vocab(vocab_map):
  vocab = []
  for idx in vocab_map:
    vocab.append(vocab_map[idx])
  return vocab


def softmax(x):
  m = np.expand_dims(np.max(x, axis=-1), -1)
  e = np.exp(x - m)
  return e / np.expand_dims(e.sum(axis=-1), -1)


def greedy_decoder(logits, vocab, merge=True):
  s = ''
  c = ''
  spaces = '#'
  for i in range(logits.shape[0]):
    max_idx = np.argmax(logits[i])
    if max_idx == len(vocab):
      continue
    c_i = vocab[max_idx]
    if merge and c_i == c:
      continue
    s += c_i
    c = c_i
    if c == " ":
      spaces += "{}#".format(i)
  return spaces, s


logits, vocab_map = load_dump(args.logits)
files, lms = load_labels(args.labels)
vocab = load_vocab(vocab_map)
infer_preds = np.empty(shape=(len(files), 3), dtype=object)
for idx, f in enumerate(files):
  key = f.replace("Book_1", "Book 1")
  probs = softmax(logits[key])
  if args.mode == "greedy":
    spaces, text = greedy_decoder(probs, vocab)
    infer_preds[idx, 0] = f
    infer_preds[idx, 1] = text
    infer_preds[idx, 2] = spaces
  else:
    scorer = Scorer(args.alpha, args.beta, model_path=lms[idx], vocabulary=vocab)
    res = ctc_beam_search_decoder(probs, vocab,
                                  beam_size=args.beam_width,
                                  ext_scoring_func=scorer)
    spaces, text = [v for v in zip(*res)]
    infer_preds[idx, 0] = f
    infer_preds[idx, 1] = text[0]
    infer_preds[idx, 2] = spaces[0]
csv_df = pd.DataFrame.from_records(infer_preds, columns=['File', 'Transcript', 'Timestamps'])
csv_df.to_csv(args.infer_output_file)