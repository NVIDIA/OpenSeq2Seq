# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
#from open_seq2seq.data.text2text.text2text import SpecialTextTokens

import argparse
import sentencepiece as spm


vocab_size = 32768

def train_tokenizer_model(args):
  print("========> Training tokenizer model")
  vocab_size = args.vocab_size
  model_prefix = args.model_prefix
  input_file = args.text_input

  spm.SentencePieceTrainer.Train(
    "--input={0} --model_type=bpe --model_prefix={1} --vocab_size={2} --pad_id={3} --eos_id={4} --bos_id={5} --unk_id={6}"
      .format(input_file,
              model_prefix, vocab_size, 0, # PAD. TODO: these should not be hardcoded
              1, 2, # EOS, SID
              3) # UNK 
  )

def tokenize(args):
  print("========> Using tokenizer model")
  model_prefix1 = args.model_prefix1
  model_prefix2 = args.model_prefix2
  input_file1 = args.text_input1
  input_file2 = args.text_input2
  tokenized_output1 = args.tokenized_output1
  tokenized_output2 = args.tokenized_output2

  sp1 = spm.SentencePieceProcessor()
  sp1.Load(model_prefix1+".model")
  sp2 = spm.SentencePieceProcessor()
  sp2.Load(model_prefix2 + ".model")

  ind = 0
  with open(input_file1, 'r') as file1, open(input_file2, 'r') as file2:
    with open(tokenized_output1, 'w') as ofile1, open(tokenized_output2, 'w') as ofile2:
      while True: # YaY!
        _src_raw = file1.readline()
        _tgt_raw = file2.readline()

        if not _src_raw or not _tgt_raw:
          break

        src_raw = _src_raw.strip()
        tgt_raw = _tgt_raw.strip()

        try:
          encoded_src_list = sp1.EncodeAsPieces(src_raw)
          encoded_tgt_list = sp2.EncodeAsPieces(tgt_raw)
        except:
          continue

        encoded_src = ' '.join([w for w in encoded_src_list])
        encoded_tgt = ' '.join([w for w in encoded_tgt_list])

        ofile1.write(encoded_src + "\n")
        ofile2.write(encoded_tgt + "\n")
        ind += 1

def detokenize(args):
  print("========> Detokenizing")
  model_prefix = args.model_prefix
  sp = spm.SentencePieceProcessor()
  sp.Load(model_prefix+".model")
  input_file = args.text_input
  output_file = args.decoded_output
  with open(output_file, 'w') as otpt:
    with open(input_file, 'r') as inpt:
      for line in inpt:
        decoded_line = sp.DecodePieces(line.split(" "))
        otpt.write(decoded_line)

def main():
  parser = argparse.ArgumentParser(description='Input Parameters')
  parser.add_argument("--text_input",
                      help="Path to text")
  parser.add_argument("--decoded_output",
                      help="Path were to save decoded output during decoding")
  parser.add_argument("--text_input1",
                      help="Path to src text when tokenizing")
  parser.add_argument("--text_input2",
                      help="Path to tgt text when tokenizing")
  parser.add_argument("--tokenized_output1",
                      help="Path to tokenized src text results")
  parser.add_argument("--tokenized_output2",
                      help="Path to tokenized tgt text results")
  parser.add_argument("--model_prefix",
                      help="model prefix")
  parser.add_argument("--model_prefix1",
                      help="model prefix for src when tokenizing")
  parser.add_argument("--model_prefix2",
                      help="model prefix for tgt when tokenizing")
  parser.add_argument('--vocab_size', type=int, default=vocab_size,
                      help='Vocabulary size')
  parser.add_argument('--mode', required=True,
                      help='train, tokenize or detokenize')
  args, unknown = parser.parse_known_args()
  if args.mode == "train":
    train_tokenizer_model(args)
  elif args.mode == "tokenize":
    tokenize(args)
  elif args.mode == "detokenize":
    detokenize(args)
  else:
    raise ValueError('Unknown mode: {0}', args.mode)


if __name__ == '__main__':
  main()
