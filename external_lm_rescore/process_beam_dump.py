# coding: utf-8
import sys
sys.path.append("./transformerxl")
sys.path.append("./transformerxl/utils")
import argparse
from typing import List
import torch
import random
from utils.vocabulary import Vocab
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser(
  description='Process OS2S output with external LM')
parser.add_argument('--beam_dump', type=str, default='',
                    help='path to OS2S beam dump')
parser.add_argument('--beam_dump_with_lm', type=str, default='',
                    help='this is where beam dump will be augmented '
                         'with LM score')
parser.add_argument('--model', type=str, default='',
                    help='path to neural language model')
parser.add_argument('--vocab', type=str, default='',
                    help='path to vocabluary')
parser.add_argument('--reference', type=str, default='',
                    help='path to reference against which to compare')

args = parser.parse_args()


def levenshtein(a, b):
  """Calculates the Levenshtein distance between a and b.
  The code was copied from: http://hetland.org/coding/python/levenshtein.py
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

def score_fun_linear(s1, s2, s3, s4):
  return s4 + s1


class Scorer:
  def __init__(self, model, path_2_vocab, score_fn=score_fun_linear):
    self._model = model
    self._model.eval()
    self._model.crit.keep_order=True
    self._vocab = Vocab(vocab_file=path_2_vocab)
    self._vocab.build_vocab()
    self._score_fn = score_fn

    print('---->>> Testing Model.')
    self.test_model(candidates=['they had one night in which to prepare for deach',
                                'they had one night in which to prepare for death',
                                'i hate school', 'i love school',
                                'the fox jumps on a grass',
                                'the crox jump a la glass'])
    print('---->>> Done testing model')


  @staticmethod
  def chunks(l, n):
    for i in range(0, len(l), n):
      yield l[i:i + n]


  def nlm_compute(self, candidates_full, batch_size=256):
    results = torch.zeros(len(candidates_full))
    with torch.no_grad():
      for j, candidates in enumerate(self.chunks(candidates_full, batch_size)):
        sents = self._vocab.encode_sents(
          [['<S>'] + string.strip().lower().split() + ['<S>'] for string in candidates])
        seq_lens = torch.tensor([x.shape[0] for x in sents], dtype=torch.long)
        sents_th = torch.zeros(seq_lens.max(), seq_lens.shape[0],dtype=torch.long).cuda()
        for i, sent in enumerate(sents):
          sents_th[:seq_lens[i], i] = sent
       
        mems = tuple()
        ret = self._model(sents_th[:-1], sents_th[1:], *mems)
        max_len = seq_lens.max()-1
        mask = torch.arange(max_len).expand(seq_lens.shape[0], max_len) >= seq_lens.unsqueeze(1)-1
        result = -1 * ret[0].masked_fill(mask.transpose(0,1).to("cuda"), 0).sum(dim=0)
        results[j*batch_size:j*batch_size + len(result)] = result
    return results
  

  def test_model(self, candidates):
    for item in zip(list(self.nlm_compute(candidates).cpu().detach().numpy()), candidates):
      print("{0} ---- {1}".format(item[0], item[1]))


  def chose_best_candidate(self, candidates: List) -> str:
    candidates_t = [c[3] for c in candidates]
    nln_scores = self.nlm_compute(candidates_t)
    candidate = candidates[0][3]
    score = -100000000000.0
    for i in range(len(candidates)):
      s1 = candidates[i][0]
      s2 = candidates[i][1]
      s3 = candidates[i][2]
      s4 = nln_scores[i].item()
      new_score = self._score_fn(s1, s2, s3, s4)
      if new_score > score:
        candidate = candidates[i][3]
        score = new_score
    return (candidate, nln_scores)

def main():
  if args.beam_dump == '':
    print("Please provide path to OS2S beam dump")
    exit(1)

  with open(args.model, 'rb') as f:
    #rnn_lm = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    #rnn_lm.rnn.flatten_parameters()
    lm = torch.load(f)
  #lm = InferenceModel(rnn_lm)
  scorer = Scorer(lm, args.vocab)
  #scorer = Scorer(rnn_lm, args.vocab)

  reference_strings = []
  first = True
  with open(args.reference, 'r') as inpr:
    for line in inpr:
      if first: # skip header
        first = False
        continue
      reference_strings.append(line.split(',')[2])

  print('Read {0} reference lines from {1}'.format(len(reference_strings),
                                                   args.reference))

  scores = 0
  words = 0
  counter = 0
  with open(args.beam_dump, 'r') as inpf:
    with open(args.beam_dump_with_lm, 'w') as outf:
      candidate_list = []
      for line in inpf:
        sline = line.strip()
        # sample begin
        if sline == "B=>>>>>>>>":
          candidate_list = []
        # sample end
        elif sline == "E=>>>>>>>>":
          if counter % 100 == 0:
            print("Processed {0} candidates".format(counter))
          candidate, nlm_scores = scorer.chose_best_candidate(candidate_list)
          words += len(reference_strings[counter].split())
          scores += levenshtein(reference_strings[counter].split(),
                                candidate.split())
          counter += 1
          # output augmented scores:
          outf.write("B=>>>>>>>>\n")
          assert(len(nlm_scores) == len(candidate_list))
          for i in range(len(nlm_scores)):
            outf.write("\t".join(
              [str(nlm_scores[i].item())] + [str(t) for t in
                                             list(candidate_list[i])]) + "\n")
          outf.write("E=>>>>>>>>\n")

        else:
          sparts = sline.split()
          s1 = float(sparts[0])
          s2 = float(sparts[1])
          s3 = float(sparts[2])
          c = ' '.join(sparts[3:])
          candidate_list.append((s1, s2, s3, c))
  print("WER: {0} after processing {1} predictions".format((scores*1.0)/words,
                                                           counter))


if __name__ == "__main__":
    main()
