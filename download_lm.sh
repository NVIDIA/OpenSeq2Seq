set -e
wget -O lm.binary https://github.com/mozilla/DeepSpeech/blob/master/data/lm/lm.binary?raw=true
wget https://github.com/mozilla/DeepSpeech/raw/master/data/lm/trie
mkdir language_model
mv lm.binary trie language_model
