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
# generate trie for prefix check
if [ ! -f "librispeech-vocab.txt" ]; then
  wget http://www.openslr.org/resources/11/librispeech-vocab.txt
fi
tr '[:upper:]' '[:lower:]' < librispeech-vocab.txt > trie_vocab.txt
../ctc_decoder_with_lm/generate_trie ../open_seq2seq/test_utils/toy_speech_data/vocab.txt ./4-gram.binary ./trie_vocab.txt ./trie.binary
