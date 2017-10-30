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
cd "$SCRIPT_DIR"
rm -rf "$TEMP_DIR"
