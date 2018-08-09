#!/usr/bin/env bash
set -e
if [ ! -d "language_model" ]; then
  mkdir language_model
fi
cd language_model
if [ ! -f "lm.binary" ]; then
  wget https://github.com/mozilla/DeepSpeech/raw/master/data/lm/lm.binary
fi
if [ ! -f "trie" ]; then
  wget https://github.com/mozilla/DeepSpeech/raw/master/data/lm/trie
fi
