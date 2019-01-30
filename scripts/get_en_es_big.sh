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

OUTPUT_DIR="${1:-wmt16_es_en}"
VOCAB_SIZE=32768
echo "Writing to ${OUTPUT_DIR}. To change this, pass the desired output dir as the first parameter."

OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"

if [ ! -d $OUTPUT_DIR_DATA ]
then 

mkdir -p $OUTPUT_DIR_DATA

echo "Downloading ParaCrawl Corpus v1.2. This may take a while..."
curl -o ${OUTPUT_DIR_DATA}/paracrawl-v1-2-es-en.tgz \
  https://s3.amazonaws.com/web-language-models/paracrawl/release1/paracrawl-release1.en-es.zipporah0-dedup-clean.tgz

echo "Downloading Europarl v7. This may take a while..."
curl -o ${OUTPUT_DIR_DATA}/europarl-v7-es-en.tgz \
  http://www.statmt.org/europarl/v7/es-en.tgz

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
mkdir -p "${OUTPUT_DIR_DATA}/paracrawl-v1-2-es-en"
tar -xvzf "${OUTPUT_DIR_DATA}/paracrawl-v1-2-es-en.tgz" -C "${OUTPUT_DIR_DATA}/paracrawl-v1-2-es-en"

mkdir -p "${OUTPUT_DIR_DATA}/europarl-v7-es-en"
tar -xvzf "${OUTPUT_DIR_DATA}/europarl-v7-es-en.tgz" -C "${OUTPUT_DIR_DATA}/europarl-v7-es-en"
mkdir -p "${OUTPUT_DIR_DATA}/common-crawl"
tar -xvzf "${OUTPUT_DIR_DATA}/common-crawl.tgz" -C "${OUTPUT_DIR_DATA}/common-crawl"
mkdir -p "${OUTPUT_DIR_DATA}/nc-v11"
tar -xvzf "${OUTPUT_DIR_DATA}/nc-v11.tgz" -C "${OUTPUT_DIR_DATA}/nc-v11"
mkdir -p "${OUTPUT_DIR_DATA}/dev"
tar -xvzf "${OUTPUT_DIR_DATA}/dev.tgz" -C "${OUTPUT_DIR_DATA}/dev"
mkdir -p "${OUTPUT_DIR_DATA}/test"
tar -xvzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"

fi


#  "${OUTPUT_DIR_DATA}/nc-v11/training-parallel-nc-v11/news-commentary-v11.es-en.en" \
# Concatenate Training data
cat "${OUTPUT_DIR_DATA}/europarl-v7-es-en/europarl-v7.es-en.en" \
  "${OUTPUT_DIR_DATA}/paracrawl-v1-2-es-en/paracrawl-release1.en-es.zipporah0-dedup-clean.en" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.es-en.en" \
  > "${OUTPUT_DIR}/train.en"
wc -l "${OUTPUT_DIR}/train.en"

cat "${OUTPUT_DIR_DATA}/europarl-v7-es-en/europarl-v7.es-en.es" \
  "${OUTPUT_DIR_DATA}/paracrawl-v1-2-es-en/paracrawl-release1.en-es.zipporah0-dedup-clean.es" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.es-en.es" \
  > "${OUTPUT_DIR}/train.es"
wc -l "${OUTPUT_DIR}/train.es"


# Concatenate Training data
cat "${OUTPUT_DIR_DATA}/europarl-v7-es-en/europarl-v7.es-en.en" \
  "${OUTPUT_DIR_DATA}/paracrawl-v1-2-es-en/paracrawl-release1.en-es.zipporah0-dedup-clean.en" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.es-en.en" \
  > "${OUTPUT_DIR}/train.small.en"
wc -l "${OUTPUT_DIR}/train.small.en"

cat "${OUTPUT_DIR_DATA}/europarl-v7-es-en/europarl-v7.es-en.es" \
  "${OUTPUT_DIR_DATA}/paracrawl-v1-2-es-en/paracrawl-release1.en-es.zipporah0-dedup-clean.es" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.es-en.es" \
  > "${OUTPUT_DIR}/train.small.es"
wc -l "${OUTPUT_DIR}/train.small.es"


# Get Eval Data
sacrebleu -t wmt13 -l en-es --echo src > ${OUTPUT_DIR}/wmt13-en-es.src
sacrebleu -t wmt13 -l en-es --echo ref > ${OUTPUT_DIR}/wmt13-en-es.ref

# Get Test Data
# sacrebleu -t wmt14 -l en-es --echo src > ${OUTPUT_DIR}/wmt14-en-es.src
# sacrebleu -t wmt14 -l en-es --echo ref > ${OUTPUT_DIR}/wmt14-en-es.ref

# Clean data
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/training/clean-corpus-n.perl
chmod +x clean-corpus-n.perl
./clean-corpus-n.perl ${OUTPUT_DIR}/train en es ${OUTPUT_DIR}/train.clean 1 80

./clean-corpus-n.perl ${OUTPUT_DIR}/train.small en es ${OUTPUT_DIR}/train.small.clean 1 80

echo 'Shuffling'
shuf --random-source=${OUTPUT_DIR}/train.clean.en ${OUTPUT_DIR}/train.clean.en > ${OUTPUT_DIR}/train.clean.en.shuffled
shuf --random-source=${OUTPUT_DIR}/train.clean.en ${OUTPUT_DIR}/train.clean.es > ${OUTPUT_DIR}/train.clean.es.shuffled
cat ${OUTPUT_DIR}/train.clean.en.shuffled ${OUTPUT_DIR}/train.clean.es.shuffled > ${OUTPUT_DIR}/train.clean.en-es.shuffled.common

shuf --random-source=${OUTPUT_DIR}/train.small.clean.en ${OUTPUT_DIR}/train.small.clean.en > ${OUTPUT_DIR}/train.small.clean.en.shuffled
shuf --random-source=${OUTPUT_DIR}/train.small.clean.en ${OUTPUT_DIR}/train.small.clean.es > ${OUTPUT_DIR}/train.small.clean.es.shuffled
cat ${OUTPUT_DIR}/train.small.clean.en.shuffled ${OUTPUT_DIR}/train.small.clean.es.shuffled > ${OUTPUT_DIR}/train.small.clean.en-es.shuffled.common


echo 'TOKENIZATION'
## Common
python tokenizer_wrapper.py  \
  --text_input=${OUTPUT_DIR}/train.clean.en-es.shuffled.common \
  --model_prefix=${OUTPUT_DIR}/m_common --vocab_size=${VOCAB_SIZE} --mode=train

# Training Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_common \
  --model_prefix2=${OUTPUT_DIR}/m_common \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/train.clean.en.shuffled \
  --text_input2=${OUTPUT_DIR}/train.clean.es.shuffled \
  --tokenized_output1=${OUTPUT_DIR}/train.clean.en.shuffled.BPE_common.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/train.clean.es.shuffled.BPE_common.32K.tok

# Eval Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_common \
  --model_prefix2=${OUTPUT_DIR}/m_common \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/wmt13-en-es.src \
  --text_input2=${OUTPUT_DIR}/wmt13-en-es.ref \
  --tokenized_output1=${OUTPUT_DIR}/wmt13-en-es.src.BPE_common.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/wmt13-en-es.ref.BPE_common.32K.tok


## Language-dependent
python tokenizer_wrapper.py  \
  --text_input=${OUTPUT_DIR}/train.clean.en.shuffled \
  --model_prefix=${OUTPUT_DIR}/m_en --vocab_size=${VOCAB_SIZE} --mode=train
python tokenizer_wrapper.py  \
  --text_input=${OUTPUT_DIR}/train.clean.es.shuffled \
  --model_prefix=${OUTPUT_DIR}/m_es --vocab_size=${VOCAB_SIZE} --mode=train

# Training Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_en \
  --model_prefix2=${OUTPUT_DIR}/m_es \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/train.clean.en.shuffled \
  --text_input2=${OUTPUT_DIR}/train.clean.es.shuffled \
  --tokenized_output1=${OUTPUT_DIR}/train.clean.en.shuffled.BPE.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/train.clean.es.shuffled.BPE.32K.tok

# Eval Set
python tokenizer_wrapper.py \
  --model_prefix1=${OUTPUT_DIR}/m_en \
  --model_prefix2=${OUTPUT_DIR}/m_es \
  --mode=tokenize \
  --text_input1=${OUTPUT_DIR}/wmt13-en-es.src \
  --text_input2=${OUTPUT_DIR}/wmt13-en-es.ref \
  --tokenized_output1=${OUTPUT_DIR}/wmt13-en-es.src.BPE.32K.tok \
  --tokenized_output2=${OUTPUT_DIR}/wmt13-en-es.ref.BPE.32K.tok

