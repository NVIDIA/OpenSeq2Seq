#!/bin/bash

set -e # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEMP_DIR="$(mktemp -d)"
OUTPUT_DIR="$SCRIPT_DIR/wmt16_en_dt"

echo "Downloading data to $OUTPUT_DIR ..."

mkdir "$OUTPUT_DIR"
export OUTPUT_DIR

cd "$TEMP_DIR"
git clone https://github.com/google/seq2seq
cd seq2seq
./bin/data/wmt16_en_de.sh


declare -a train_list_en=("train.tok.clean.bpe.32000.en"
                           "train.tok.bpe.32000.en"
                           "train.tok.clean.en"
                           "train.tok.en"
                           "train.clean.en"
                           "train.en")

declare -a train_list_de=("train.tok.clean.bpe.32000.de"
                           "train.tok.bpe.32000.de"
                           "train.tok.clean.de"
                           "train.tok.de"
                           "train.clean.de"
                           "train.de")

list_size=${#train_list_en[@]}

for (( i=0; i<${list_size}; i++ ));
do
  shuf --random-source=${OUTPUT_DIR}/${train_list_en[$i]} ${OUTPUT_DIR}/${train_list_en[$i]} > ${OUTPUT_DIR}/${train_list_en[$i]}.shuffled
  shuf --random-source=${OUTPUT_DIR}/${train_list_en[$i]} ${OUTPUT_DIR}/${train_list_de[$i]} > ${OUTPUT_DIR}/${train_list_de[$i]}.shuffled
  mv ${OUTPUT_DIR}/${train_list_en[$i]}.shuffled ${OUTPUT_DIR}/${train_list_en[$i]}
  mv ${OUTPUT_DIR}/${train_list_de[$i]}.shuffled ${OUTPUT_DIR}/${train_list_de[$i]}
done

cd "$SCRIPT_DIR"
rm -rf "$TEMP_DIR"
