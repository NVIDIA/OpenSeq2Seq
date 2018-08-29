#!/bin/bash
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by okuchaiev@nvidia.com to use different tokenizer

set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

OUTPUT_DIR="${1:-wmt16_de_en}"
VOCAB_SIZE=32768
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"
mkdir -p $OUTPUT_DIR_DATA

echo "Downloading Europarl v7. This may take a while..."
curl -o ${OUTPUT_DIR_DATA}/europarl-v7-de-en.tgz \
  http://www.statmt.org/europarl/v7/de-en.tgz

echo "Downloading Common Crawl corpus. This may take a while..."
curl -o ${OUTPUT_DIR_DATA}/common-crawl.tgz \
  http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz

echo "Downloading News Commentary v11. This may take a while..."
curl -o ${OUTPUT_DIR_DATA}/nc-v11.tgz \
  http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz

echo "Downloading dev/test sets"
curl -o ${OUTPUT_DIR_DATA}/dev.tgz \
  http://data.statmt.org/wmt16/translation-task/dev.tgz
curl -o ${OUTPUT_DIR_DATA}/test.tgz \
  http://data.statmt.org/wmt16/translation-task/test.tgz

# Extract everything
echo "Extracting all files..."
mkdir -p "${OUTPUT_DIR_DATA}/europarl-v7-de-en"
tar -xvzf "${OUTPUT_DIR_DATA}/europarl-v7-de-en.tgz" -C "${OUTPUT_DIR_DATA}/europarl-v7-de-en"
mkdir -p "${OUTPUT_DIR_DATA}/common-crawl"
tar -xvzf "${OUTPUT_DIR_DATA}/common-crawl.tgz" -C "${OUTPUT_DIR_DATA}/common-crawl"
mkdir -p "${OUTPUT_DIR_DATA}/nc-v11"
tar -xvzf "${OUTPUT_DIR_DATA}/nc-v11.tgz" -C "${OUTPUT_DIR_DATA}/nc-v11"
mkdir -p "${OUTPUT_DIR_DATA}/dev"
tar -xvzf "${OUTPUT_DIR_DATA}/dev.tgz" -C "${OUTPUT_DIR_DATA}/dev"
mkdir -p "${OUTPUT_DIR_DATA}/test"
tar -xvzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"

# Concatenate Training data
cat "${OUTPUT_DIR_DATA}/europarl-v7-de-en/europarl-v7.de-en.en" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.de-en.en" \
  "${OUTPUT_DIR_DATA}/nc-v11/training-parallel-nc-v11/news-commentary-v11.de-en.en" \
  > "${OUTPUT_DIR}/train.en"
wc -l "${OUTPUT_DIR}/train.en"

cat "${OUTPUT_DIR_DATA}/europarl-v7-de-en/europarl-v7.de-en.de" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.de-en.de" \
  "${OUTPUT_DIR_DATA}/nc-v11/training-parallel-nc-v11/news-commentary-v11.de-en.de" \
  > "${OUTPUT_DIR}/train.de"
wc -l "${OUTPUT_DIR}/train.de"

# Get Eval Data
sacrebleu -t wmt13 -l en-de --echo src > ${OUTPUT_DIR}/wmt13-en-de.src
sacrebleu -t wmt13 -l en-de --echo ref > ${OUTPUT_DIR}/wmt13-en-de.ref

# Get Test Data
sacrebleu -t wmt14 -l en-de --echo src > ${OUTPUT_DIR}/wmt14-en-de.src
sacrebleu -t wmt14 -l en-de --echo ref > ${OUTPUT_DIR}/wmt14-en-de.ref

# Clean data
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/training/clean-corpus-n.perl
chmod +x clean-corpus-n.perl
./clean-corpus-n.perl ${OUTPUT_DIR}/train en de ${OUTPUT_DIR}/train.clean 1 80

echo 'Shuffling'
shuf --random-source=${OUTPUT_DIR}/train.clean.en ${OUTPUT_DIR}/train.clean.en > ${OUTPUT_DIR}/train.clean.en.shuffled
shuf --random-source=${OUTPUT_DIR}/train.clean.en ${OUTPUT_DIR}/train.clean.de > ${OUTPUT_DIR}/train.clean.de.shuffled
cat ${OUTPUT_DIR}/train.clean.en.shuffled ${OUTPUT_DIR}/train.clean.de.shuffled > ${OUTPUT_DIR}/train.clean.en-de.shuffled.common

echo 'TOKENIZATION'
## Common
python tokenizer_wrapper.py  \
  --text_input=${OUTPUT_DIR}/train.clean.en-de.shuffled.common \
  --model_prefix=${OUTPUT_DIR}/m_common --vocab_size=${VOCAB_SIZE} --mode=train

# Training Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_common \
  --model_prefix2=${OUTPUT_DIR}/m_common \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/train.clean.en.shuffled \
  --text_input2=${OUTPUT_DIR}/train.clean.de.shuffled \
  --tokenized_output1=${OUTPUT_DIR}/train.clean.en.shuffled.BPE_common.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/train.clean.de.shuffled.BPE_common.32K.tok

# Eval Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_common \
  --model_prefix2=${OUTPUT_DIR}/m_common \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/wmt13-en-de.src \
  --text_input2=${OUTPUT_DIR}/wmt13-en-de.ref \
  --tokenized_output1=${OUTPUT_DIR}/wmt13-en-de.src.BPE_common.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/wmt13-en-de.ref.BPE_common.32K.tok

# Test Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_common \
  --model_prefix2=${OUTPUT_DIR}/m_common \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/wmt14-en-de.src \
  --text_input2=${OUTPUT_DIR}/wmt14-en-de.ref \
  --tokenized_output1=${OUTPUT_DIR}/wmt14-en-de.src.BPE_common.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/wmt14-en-de.ref.BPE_common.32K.tok

## Language-dependent
python tokenizer_wrapper.py  \
  --text_input=${OUTPUT_DIR}/train.clean.en.shuffled \
  --model_prefix=${OUTPUT_DIR}/m_en --vocab_size=${VOCAB_SIZE} --mode=train
python tokenizer_wrapper.py  \
  --text_input=${OUTPUT_DIR}/train.clean.de.shuffled \
  --model_prefix=${OUTPUT_DIR}/m_de --vocab_size=${VOCAB_SIZE} --mode=train

# Training Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_en \
  --model_prefix2=${OUTPUT_DIR}/m_de \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/train.clean.en.shuffled \
  --text_input2=${OUTPUT_DIR}/train.clean.de.shuffled \
  --tokenized_output1=${OUTPUT_DIR}/train.clean.en.shuffled.BPE.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/train.clean.de.shuffled.BPE.32K.tok

# Eval Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_en \
  --model_prefix2=${OUTPUT_DIR}/m_de \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/wmt13-en-de.src \
  --text_input2=${OUTPUT_DIR}/wmt13-en-de.ref \
  --tokenized_output1=${OUTPUT_DIR}/wmt13-en-de.src.BPE.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/wmt13-en-de.ref.BPE.32K.tok

# Test Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_en \
  --model_prefix2=${OUTPUT_DIR}/m_de \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/wmt14-en-de.src \
  --text_input2=${OUTPUT_DIR}/wmt14-en-de.ref \
  --tokenized_output1=${OUTPUT_DIR}/wmt14-en-de.src.BPE.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/wmt14-en-de.ref.BPE.32K.tok