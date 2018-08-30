#!/usr/bin/env bash
set -e
# This will take quite some time
pip install -r requirements.txt
echo '**********>>>> CREATE TOY DATA <<<< ************'
scripts/create_toy_data.sh
echo '**********>>>> RUNNING UNIT TESTS <<<< ************'
python -m unittest discover -s open_seq2seq -p '*_test.py'

echo '**********>>>> RUNNING SMALL Models <<<< ************'
if [ -d "log_text2text_small" ]; then
  rm -rf log_text2text_small
fi
mkdir log_text2text_small
chmod +x scripts/multi-bleu.perl

python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-RR.py \
  --mode=train_eval --logdir=log_text2text_small/RR
python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-RR.py \
  --mode=infer --logdir=log_text2text_small/RR \
  --infer_output_file=log_text2text_small/RR.out --num_gpus=1
scripts/multi-bleu.perl toy_text_data/test/target.txt < log_text2text_small/RR.out > \
  log_text2text_small/RR.BLEU
echo 'RR BLEU:'
cat log_text2text_small/RR.BLEU

python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-CC.py \
  --mode=train_eval --logdir=log_text2text_small/CC
python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-CC.py \
  --mode=infer --logdir=log_text2text_small/CC \
  --infer_output_file=log_text2text_small/CC.out --num_gpus=1
scripts/multi-bleu.perl toy_text_data/test/target.txt < log_text2text_small/CC.out > \
  log_text2text_small/CC.BLEU
echo 'CC BLEU:'
cat log_text2text_small/CC.BLEU

python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-CR.py \
  --mode=train_eval --logdir=log_text2text_small/CR
python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-CR.py \
  --mode=infer --logdir=log_text2text_small/CR \
  --infer_output_file=log_text2text_small/CR.out --num_gpus=1
scripts/multi-bleu.perl toy_text_data/test/target.txt < log_text2text_small/CR.out > \
  log_text2text_small/CR.BLEU
echo 'CR BLEU:'
cat log_text2text_small/CR.BLEU

python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-RC.py \
  --mode=train_eval --logdir=log_text2text_small/RC
python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-RC.py \
  --mode=infer --logdir=log_text2text_small/RC \
  --infer_output_file=log_text2text_small/RC.out --num_gpus=1
scripts/multi-bleu.perl toy_text_data/test/target.txt < log_text2text_small/RC.out > \
  log_text2text_small/RC.BLEU
echo 'RC BLEU:'
cat log_text2text_small/RC.BLEU

python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-TT.py \
  --mode=train_eval --logdir=log_text2text_small/TT
python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-TT.py \
  --mode=infer --logdir=log_text2text_small/TT \
  --infer_output_file=log_text2text_small/TT.out --num_gpus=1
scripts/multi-bleu.perl toy_text_data/test/target.txt < log_text2text_small/TT.out > \
  log_text2text_small/TT.BLEU
echo 'TT BLEU:'
cat log_text2text_small/TT.BLEU

