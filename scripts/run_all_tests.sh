#!/usr/bin/env bash
set -e
# This will take quite some time
pip install -r requirements.txt
echo '**********>>>> CREATE TOY DATA <<<< ************'
scripts/create_toy_data.sh
echo '**********>>>> RUNNING UNIT TESTS <<<< ************'
python -m unittest discover -s open_seq2seq -p '*_test.py'

echo '**********>>>> RUNNING SMALL Models <<<< ************'

python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-RR.py \
  --mode=train_eval --logdir=RR
python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-RR.py \
  --mode=infer --logdir=RR --infer_output_file=RR.out --num_gpus=1
scripts/multi-bleu.perl toy_text_data/test/target.txt < RR.out > RR.BLEU
echo 'RR BLEU:'
cat RR.BLEU

python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-CC.py \
  --mode=train_eval --logdir=CC
python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-CC.py \
  --mode=infer --logdir=CC --infer_output_file=CC.out --num_gpus=1
scripts/multi-bleu.perl toy_text_data/test/target.txt < CC.out > CC.BLEU
echo 'CC BLEU:'
cat CC.BLEU

python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-CR.py \
  --mode=train_eval --logdir=CR
python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-CR.py \
  --mode=infer --logdir=CR --infer_output_file=CR.out --num_gpus=1
scripts/multi-bleu.perl toy_text_data/test/target.txt < CR.out > CR.BLEU
echo 'CR BLEU:'
cat CR.BLEU

python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-RC.py \
  --mode=train_eval --logdir=RC
python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-RC.py \
  --mode=infer --logdir=RC --infer_output_file=RC.out --num_gpus=1
scripts/multi-bleu.perl toy_text_data/test/target.txt < RC.out > RC.BLEU
echo 'RC BLEU:'
cat RC.BLEU

python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-TT.py \
  --mode=train_eval --logdir=TT
python run.py --config_file=example_configs/text2text/toy-reversal/nmt-reversal-TT.py \
  --mode=infer --logdir=TT --infer_output_file=TT.out --num_gpus=1
scripts/multi-bleu.perl toy_text_data/test/target.txt < TT.out > TT.BLEU
echo 'TT BLEU:'
cat TT.BLEU
