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

# Clone Moses
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
fi

# Convert SGM files
# Convert newstest2014 data into raw text format
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2014-deen-src.de.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2014.de
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2014-deen-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2014.en

# Convert newstest2015 data into raw text format
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2015-deen-src.de.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2015.de
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2015-deen-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2015.en

# Convert newstest2016 data into raw text format
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test/newstest2016-deen-src.de.sgm \
  > ${OUTPUT_DIR_DATA}/test/test/newstest2016.de
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test/newstest2016-deen-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/test/test/newstest2016.en

# Copy dev/test data to output dir
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest20*.de ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest20*.en ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/test/test/newstest20*.de ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/test/test/newstest20*.en ${OUTPUT_DIR}

sacrebleu -t wmt14 -l en-de --echo src > ${OUTPUT_DIR}/wmt14-en-de.src
sacrebleu -t wmt14 -l en-de --echo ref > ${OUTPUT_DIR}/wmt14-en-de.ref


# Shuffle training set
declare -a train_list_en=("train.en")
declare -a train_list_de=("train.de")
#
list_size=${#train_list_en[@]}
#
for (( i=0; i<${list_size}; i++ ));
do
   shuf --random-source=${OUTPUT_DIR}/${train_list_en[$i]} ${OUTPUT_DIR}/${train_list_en[$i]} > ${OUTPUT_DIR}/${train_list_en[$i]}.shuffled
   shuf --random-source=${OUTPUT_DIR}/${train_list_en[$i]} ${OUTPUT_DIR}/${train_list_de[$i]} > ${OUTPUT_DIR}/${train_list_de[$i]}.shuffled
   cat ${OUTPUT_DIR}/${train_list_en[$i]}.shuffled ${OUTPUT_DIR}/${train_list_de[$i]}.shuffled > ${OUTPUT_DIR}/${train_list_en[$i]}_${train_list_de[$i]}.shuffled.common

   # Common
   python tokenizer_wrapper.py  \
   --text_input=${OUTPUT_DIR}/${train_list_en[$i]}_${train_list_de[$i]}.shuffled.common \
   --model_prefix=${OUTPUT_DIR}/m_common --vocab_size=32768 --mode=train

   # Training Set
   python tokenizer_wrapper.py \
    --model_prefix1=${OUTPUT_DIR}/m_common \
    --model_prefix2=${OUTPUT_DIR}/m_common \
    --mode=tokenize \
    --text_input1=${OUTPUT_DIR}/${train_list_en[$i]}.shuffled \
    --text_input2=${OUTPUT_DIR}/${train_list_de[$i]}.shuffled \
    --tokenized_output1=${OUTPUT_DIR}/${train_list_en[$i]}.shuffled.BPE_common.32K.tok \
    --tokenized_output2=${OUTPUT_DIR}/${train_list_de[$i]}.shuffled.BPE_common.32K.tok

   # Eval Set
   python tokenizer_wrapper.py \
    --model_prefix1=${OUTPUT_DIR}/m_common \
    --model_prefix2=${OUTPUT_DIR}/m_common \
    --mode=tokenize \
    --text_input1=${OUTPUT_DIR}/newstest2013.en \
    --text_input2=${OUTPUT_DIR}/newstest2013.de \
    --tokenized_output1=${OUTPUT_DIR}/newstest2013.en.BPE_common.32K.tok \
    --tokenized_output2=${OUTPUT_DIR}/newstest2013.de.BPE_common.32K.tok

   # Test Set1
   python tokenizer_wrapper.py \
    --model_prefix1=${OUTPUT_DIR}/m_common \
    --model_prefix2=${OUTPUT_DIR}/m_common \
    --mode=tokenize \
    --text_input1=${OUTPUT_DIR}/newstest2014.en \
    --text_input2=${OUTPUT_DIR}/newstest2014.de \
    --tokenized_output1=${OUTPUT_DIR}/newstest2014.en.BPE_common.32K.tok \
    --tokenized_output2=${OUTPUT_DIR}/newstest2014.de.BPE_common.32K.tok

   # Test Set2
   python tokenizer_wrapper.py \
    --model_prefix1=${OUTPUT_DIR}/m_common \
    --model_prefix2=${OUTPUT_DIR}/m_common \
    --mode=tokenize \
    --text_input1=${OUTPUT_DIR}/wmt14-en-de.src \
    --text_input2=${OUTPUT_DIR}/wmt14-en-de.ref \
    --tokenized_output1=${OUTPUT_DIR}/wmt14-en-de.src.BPE_common.32K.tok \
    --tokenized_output2=${OUTPUT_DIR}/wmt14-en-de.ref.BPE_common.32K.tok


   # Language-dependent
   python tokenizer_wrapper.py \
   --text_input=${OUTPUT_DIR}/${train_list_de[$i]}.shuffled \
   --model_prefix=${OUTPUT_DIR}/m_de --vocab_size=32768 --mode=train

   python tokenizer_wrapper.py \
   --text_input=${OUTPUT_DIR}/${train_list_en[$i]}.shuffled \
   --model_prefix=${OUTPUT_DIR}/m_en --vocab_size=32768 --mode=train

   python tokenizer_wrapper.py \
    --model_prefix1=${OUTPUT_DIR}/m_en \
    --model_prefix2=${OUTPUT_DIR}/m_de \
    --mode=tokenize \
    --text_input1=${OUTPUT_DIR}/${train_list_en[$i]}.shuffled \
    --text_input2=${OUTPUT_DIR}/${train_list_de[$i]}.shuffled \
    --tokenized_output1=${OUTPUT_DIR}/${train_list_en[$i]}.shuffled.BPE.32K.tok \
    --tokenized_output2=${OUTPUT_DIR}/${train_list_de[$i]}.shuffled.BPE.32K.tok

   # Eval Set
   python tokenizer_wrapper.py \
    --model_prefix1=${OUTPUT_DIR}/m_en \
    --model_prefix2=${OUTPUT_DIR}/m_de \
    --mode=tokenize \
    --text_input1=${OUTPUT_DIR}/newstest2013.en \
    --text_input2=${OUTPUT_DIR}/newstest2013.de \
    --tokenized_output1=${OUTPUT_DIR}/newstest2013.en.BPE.32K.tok \
    --tokenized_output2=${OUTPUT_DIR}/newstest2013.de.BPE.32K.tok

   # Test Set1
   python tokenizer_wrapper.py \
    --model_prefix1=${OUTPUT_DIR}/m_en \
    --model_prefix2=${OUTPUT_DIR}/m_de \
    --mode=tokenize \
    --text_input1=${OUTPUT_DIR}/newstest2014.en \
    --text_input2=${OUTPUT_DIR}/newstest2014.de \
    --tokenized_output1=${OUTPUT_DIR}/newstest2014.en.BPE.32K.tok \
    --tokenized_output2=${OUTPUT_DIR}/newstest2014.de.BPE.32K.tok

   # Test Set2
   python tokenizer_wrapper.py \
    --model_prefix1=${OUTPUT_DIR}/m_en \
    --model_prefix2=${OUTPUT_DIR}/m_de \
    --mode=tokenize \
    --text_input1=${OUTPUT_DIR}/wmt14-en-de.src \
    --text_input2=${OUTPUT_DIR}/wmt14-en-de.ref \
    --tokenized_output1=${OUTPUT_DIR}/wmt14-en-de.src.BPE.32K.tok \
    --tokenized_output2=${OUTPUT_DIR}/wmt14-en-de.ref.BPE.32K.tok

done

