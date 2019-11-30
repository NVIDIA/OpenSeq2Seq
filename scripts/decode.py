'''
Interface to Baidu's CTC decoders
from https://github.com/PaddlePaddle/DeepSpeech/decoders/swig
'''

import argparse
import time
import pickle
import numpy as np

from ctc_decoders import Scorer
from ctc_decoders import ctc_greedy_decoder
from ctc_decoders import ctc_beam_search_decoder_batch, ctc_beam_search_decoder
from collections import defaultdict
import multiprocessing


parser = argparse.ArgumentParser(
    description='CTC decoding and tuning with LM rescoring'
)
parser.add_argument('--mode',
    help='either \'eval\' (default) or \'infer\'',
    default='eval'
)
parser.add_argument('--infer_output_file',
    help='output CSV file for \'infer\' mode',
    required=False
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
parser.add_argument('--lm',
    help='KenLM binary file',
    required=True
)
parser.add_argument('--vocab',
    help='vocab file with characters (alphabet)',
    required=True
)
parser.add_argument('--alpha', type=float,
    help='value of LM weight',
    required=True
)
parser.add_argument('--alpha_max', type=float,
    help='maximum value of LM weight (for a grid search in \'eval\' mode)',
    required=False
)
parser.add_argument('--alpha_step', type=float,
    help='step for LM weight\'s tuning in \'eval\' mode',
    required=False, default=0.1
)
parser.add_argument('--beta', type=float,
    help='value of word count weight',
    required=True
)
parser.add_argument('--beta_max', type=float,
    help='maximum value of word count weight (for a grid search in \
      \'eval\' mode',
    required=False
)
parser.add_argument('--beta_step', type=float,
    help='step for word count weight\'s tuning in \'eval\' mode',
    required=False, default=0.1
)
parser.add_argument('--beam_width', type=int,
    help='beam width for beam search decoder',
    required=False, default=128
)
parser.add_argument('--dump_all_beams_to', 
    help='filename to dump all beams in eval mode for debug purposes',
    required=False, default='')
args = parser.parse_args()

if args.alpha_max is None:
  args.alpha_max = args.alpha
# include alpha_max in tuning range
args.alpha_max += args.alpha_step/10.0

if args.beta_max is None:
  args.beta_max = args.beta
# include beta_max in tuning range
args.beta_max += args.beta_step/10.0

num_cpus = multiprocessing.cpu_count()

def levenshtein(a, b):
  """Calculates the Levenshtein distance between a and b.
  The code was taken from: http://hetland.org/coding/python/levenshtein.py
  """
  n, m = len(a), len(b)
  if n > m:
    # Make sure n <= m, to use O(min(n,m)) space
    a, b = b, a
    n, m = m, n
  current = list(range(n + 1))
  for i in range(1, m + 1):
    previous, current = current, [i] + [0] * n
    for j in range(1, n + 1):
      add, delete = previous[j] + 1, current[j - 1] + 1
      change = previous[j - 1]
      if a[j - 1] != b[i - 1]:
        change = change + 1
      current[j] = min(add, delete, change)
  return current[n]


def load_dump(pickle_file):
  with open(pickle_file, 'rb') as f:
    data = pickle.load(f, encoding='bytes')
  return data


def get_logits(data, labels):
  '''
  Get logits from pickled data.
  There are two versions of pickle file (and data):
  1. raw logits NumPy array
  2. dictionary with logits and additional meta information
  '''
  if isinstance(data, np.ndarray):
    # convert NumPy array to dict format
    logits = {}
    for idx, line in enumerate(labels):
      audio_filename = line[0]
      logits[audio_filename] = data[idx]
  else:
    logits = data['logits']
  return logits


def load_labels(csv_file):
  labels = np.loadtxt(csv_file, skiprows=1, delimiter=',', dtype=str)
  return labels

    
def load_vocab(vocab_file):
  vocab = []
  with open(vocab_file, 'r') as f:
    for line in f:
      vocab.append(line[0])
  vocab.append('_')
  return vocab


def greedy_decoder(logits, vocab, merge=True):
  s = ''
  c = ''
  for i in range(logits.shape[0]):
    c_i = vocab[np.argmax(logits[i])]
    if merge and c_i == c:
      continue 
    s += c_i
    c = c_i
  if merge:
    s = s.replace('_', '')
  return s


def softmax(x):
  m = np.expand_dims(np.max(x, axis=-1), -1)
  e = np.exp(x - m)
  return e / np.expand_dims(e.sum(axis=-1), -1)


def evaluate_wer(logits, labels, vocab, decoder):
  eval_start=time.time()
  print("evaluation started at   ",eval_start)
  total_dist = 0.0
  total_count = 0.0
  wer_per_sample = np.empty(shape=len(labels))
    
  empty_preds = 0
  for idx, line in enumerate(labels):
    audio_filename = line[0]
    label = line[-1]
    pred = decoder(logits[audio_filename], vocab)
    dist = levenshtein(label.lower().split(), pred.lower().split())
    if pred=='':
      empty_preds += 1
    total_dist += dist
    total_count += len(label.split())
    wer_per_sample[idx] = dist / len(label.split())
  print('# empty preds: {}'.format(empty_preds))
  wer = total_dist / total_count
  eval_end=time.time()
  print("evaluation took %s time"%(eval_end-eval_start))
  return wer, wer_per_sample

def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

data_load_start=time.time()
data = load_dump(args.logits)
labels = load_labels(args.labels)
logits = get_logits(data, labels)
vocab = load_vocab(args.vocab)
vocab[-1] = '_'
data_load_end=time.time()
print("Data loading took %s seconds" %(data_load_end-data_load_start) )
probs_batch = []
for line in labels:
  audio_filename = line[0]
  probs_batch.append(softmax(logits[audio_filename]))
batch_prob_end=time.time()
print("Batch logit loading took %s seconds" %(batch_prob_end-data_load_end) )

if args.mode == 'eval':
  eval_start=time.time()
  wer, _ = evaluate_wer(logits, labels, vocab, greedy_decoder)
  print('Greedy WER = {:.4f}'.format(wer))
  best_result = {'wer': 1e6, 'alpha': 0.0, 'beta': 0.0, 'beams': None} 
  for alpha in np.arange(args.alpha, args.alpha_max, args.alpha_step):
    for beta in np.arange(args.beta, args.beta_max, args.beta_step):
      scorer = Scorer(alpha, beta, model_path=args.lm, vocabulary=vocab[:-1])
      print("scorer complete")
      probs_batch_list = list(divide_chunks(probs_batch, 500))
      res=[]
      for  probs_batch in probs_batch_list:
        f=time.time()
        result = ctc_beam_search_decoder_batch(probs_batch, vocab[:-1], 
                                            beam_size=args.beam_width, 
                                            num_processes=num_cpus,
                                            ext_scoring_func=scorer)
        e=time.time()
        for j in result:
          res.append(j)
        print("500 files batched took %s time"%(e-f))
        
      total_dist = 0.0
      total_count = 0.0
      for idx, line in enumerate(labels):
        label = line[-1]
        score, text = [v for v in zip(*res[idx])]
        pred = text[0]
        dist = levenshtein(label.lower().split(), pred.lower().split())
        total_dist += dist
        total_count += len(label.split())
      wer = total_dist / total_count
      if wer < best_result['wer']:
        best_result['wer'] = wer
        best_result['alpha'] = alpha
        best_result['beta'] = beta
        best_result['beams'] = res
      print('alpha={:.2f}, beta={:.2f}: WER={:.4f}'.format(alpha, beta, wer))
  print('BEST: alpha={:.2f}, beta={:.2f}, WER={:.4f}'.format(
        best_result['alpha'], best_result['beta'], best_result['wer']))
  eval_end=time.time()
  print("evaluation took %s seconds",eval_end-eval_start)  
  if args.dump_all_beams_to:
   with open(args.dump_all_beams_to, 'w') as f:
     for beam in best_result['beams']:
       f.write('B=>>>>>>>>\n')
       for pred in beam:
         f.write('{} 0.0 0.0 {}\n'.format(pred[0], pred[1]))
       f.write('E=>>>>>>>>\n')

elif args.mode == 'greedy':
    print("Greedy Mode")
    greedy_preds = np.empty(shape=(len(labels), 2), dtype=object)
    for idx, line in enumerate(labels):
        filename = line[0]
        greedy_preds[idx, 0] = filename
        greedy_preds[idx, 1] = greedy_decoder(logits[filename], vocab)

    np.savetxt(args.infer_output_file, greedy_preds, fmt='%s', delimiter=',',
              header='wav_filename,greedy')
    

elif args.mode == 'infer':
    print("Inference Mode")
    infer_start=time.time()
    scorer = Scorer(args.alpha, args.beta, model_path=args.lm, vocabulary=vocab[:-1])

    probs_batch_list = list(divide_chunks(probs_batch, 500))
    res=[]
    for  probs_batch in probs_batch_list:
      f=time.time()
      result = ctc_beam_search_decoder_batch(probs_batch, vocab[:-1], 
                                          beam_size=args.beam_width, 
                                          num_processes=num_cpus,
                                          ext_scoring_func=scorer)
      e=time.time()

      for j in result:
        res.append(j)

      print("500 files batched took %s time"%(e-f))

    infer_preds = np.empty(shape=(len(labels), 3), dtype=object)
    for idx, line in enumerate(labels):
      filename = line[0]
      score, text = [v for v in zip(*res[idx])]
      infer_preds[idx, 0] = filename
      infer_preds[idx, 1] = text[0]
      #Greedy
      infer_preds[idx, 2] = greedy_decoder(logits[filename], vocab)
    
    infer_end=time.time()
    print("Inference took %s seconds",infer_end-infer_start)  
    np.savetxt(args.infer_output_file, infer_preds, fmt='%s', delimiter=',',header='wav_filename,lm,greedy')

