#!/usr/bin/env bash
set -e
if [ ! -d "language_model" ]; then
  mkdir language_model
fi
cd language_model
if [ ! -f "4-gram.arpa.gz" ]; then
  wget http://www.openslr.org/resources/11/4-gram.arpa.gz 
fi
gzip -d 4-gram.arpa.gz
# convert all upper case characters to lower case
tr '[:upper:]' '[:lower:]' < 4-gram.arpa > 4-gram-lower.arpa
# build a quantized array binary language model
../kenlm/build/bin/build_binary trie -q 8 -b 7 -a 256 4-gram-lower.arpa 4-gram.binary

if [ ! -x "$(command -v ctc_decoder_with_lm/generate_trie)" ]; then
  echo "INFO: Skipping trie generation, since no custom TF op based CTC decoder found."
  echo "INFO: Please use Baidu CTC decoder with this language model."
else
  # generate trie for prefix check (in TF op based decoder only)
  echo "INFO: Generating a trie for custom TF op based CTC decoder."
  if [ ! -f "librispeech-vocab.txt" ]; then
    wget http://www.openslr.org/resources/11/librispeech-vocab.txt
  fi
  tr '[:upper:]' '[:lower:]' < librispeech-vocab.txt > trie_vocab.txt
  ../ctc_decoder_with_lm/generate_trie ../open_seq2seq/test_utils/toy_speech_data/vocab.txt ./4-gram.binary ./trie_vocab.txt ./trie.binary
fi
