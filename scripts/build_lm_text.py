import pandas as pd
import os
import argparse


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Build N-gram LM model from text file')
  parser.add_argument('text', metavar='text', type=str,
                      help='text file')
  parser.add_argument('--n', type=int, help='n for n-grams', default=3)
  args = parser.parse_args()

  path_prefix, _ = os.path.splitext(args.text)
  corpus_name = args.text
  arpa_name = path_prefix + '.arpa'
  lm_name = path_prefix + '-lm.binary'

  command = 'kenlm/build/bin/lmplz --text {} --arpa {} --o {}'.format(
    corpus_name, arpa_name, args.n)
  print(command)
  os.system(command)

  command = 'kenlm/build/bin/build_binary trie -q 8 -b 7 -a 256 {} {}'.format(
    arpa_name, lm_name)
  print(command)
  os.system(command)

  command = 'ctc_decoder_with_lm/generate_trie'
  if os.path.isfile(command) and os.access(command, os.X_OK):
    trie_name = path_prefix + '-lm.trie'
    command += ' open_seq2seq/test_utils/toy_speech_data/vocab.txt {} {} {}'.format(
        lm_name, corpus_name, trie_name)
    print('INFO: Generating a trie for custom TF op based CTC decoder.')
    print(command)
    os.system(command)
  else:
    print('INFO: Skipping trie generation, since no custom TF op based CTC decoder found.')
    print('INFO: Please use Baidu CTC decoder with this language model.')

