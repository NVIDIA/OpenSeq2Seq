#!/bin/bash
#This script allows new users to run the GNMT like model on wmt16 english to deutsch dataset,
#examin the results on tensorboard and benchmark performance.
#ARGS-
#$1 arg is the desired base path for the script to write auxilary files
#$2 arg is number of gpus to use for training.
#$3 arg is the wanted batch size.
#$4 arg is the number of training steps between reports.
#$5 arg is used to specify the data-set location, if not used output variable is set to <auxilary dir>/data.
#$6 arg is number of bench samples. If set training will terminate after bench*summary_freq training steps!
set -e

ORIGIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR=${1:-"/tmp/gnmt"}
NUM_GPU=${2:-1}
BATCH_SIZE=${3:-128}
SUMMARY_FREQ=${4:-100}
OUTPUT_DIR=${5:-$BASE_DIR/data}
BENCH_SAMPLES=${6:-0}

echo 'base dir is '$BASE_DIR
echo 'data dir is '$OUTPUT_DIR

#create base directory
if [ ! -d "$BASE_DIR"  ]; then
  mkdir -p $BASE_DIR
fi

#setup the data set in the data directory
if [ ! -d "$OUTPUT_DIR"  ]; then
  git clone https://github.com/google/seq2seq $BASE_DIR/wmt_temp
  mkdir -p $OUTPUT_DIR
  cd $BASE_DIR/wmt_temp/bin/data/ #needs to cd since this uses relative locations
  source wmt16_en_de.sh
  rm -rf $BASE_DIR/wmt_temp
else
  echo 'designted data folder '"$OUTPUT_DIR"' already exists, assuming data is available'
fi

cd $ORIGIN/example_configs

cat gnmt_like.json | sed 's@\[WMT16_DATA_LOCATION\]@'"$OUTPUT_DIR"'@' |  sed 's@"num_gpus" : 1,@"num_gpus" : '"$NUM_GPU"' ,@' |  sed 's@"batch_size" : 128,@"batch_size" : '"$BATCH_SIZE"',@'  > $BASE_DIR/train_gnmt_conf.json
echo "edited train script is writen into "$BASE_DIR"/train_gnmt_conf.json"

cd ..

LOG_DIR="$BASE_DIR""/gnmt_on_""$NUM_GPU""_GPUs""_batchsize_""$BATCH_SIZE"

#clear previous run logs
if [ -d "$LOG_DIR"  ]; then
  rm -rf $LOG_DIR
fi

mkdir -p $LOG_DIR/model-eval
echo 'logs can be found at '"$LOG_DIR"

if [ $BENCH_SAMPLES -gt 0 ]; then 
  echo 'start benchmark for Nvidia flavor GNMT with '"$NUM_GPU"' GPUs, and batch size of '"$BATCH_SIZE"''
  sed '/eval/d' $BASE_DIR/train_gnmt_conf.json > $BASE_DIR/bench_gnmt_conf.json
  head $BASE_DIR/bench_gnmt_conf.json
  python bench.py --config_file=$BASE_DIR/bench_gnmt_conf.json --logdir=$LOG_DIR/ \
    --summary_frequency=$SUMMARY_FREQ --bench_steps=$BENCH_SAMPLES

else
  echo 'launching tensor-board'
  tensorboard --logdir $LOG_DIR &
  echo 'start training Nvidia flavor GNMT with '"$NUM_GPU"' GPUs, and batch size of '"$BATCH_SIZE"''
  head $BASE_DIR/train_gnmt_conf.json
  python run.py --config_file=$BASE_DIR/train_gnmt_conf.json --logdir=$LOG_DIR/ \
    --checkpoint_frequency=100000 --summary_frequency=$SUMMARY_FREQ --eval_frequency=10000
fi
