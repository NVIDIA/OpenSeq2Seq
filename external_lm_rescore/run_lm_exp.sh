#!/bin/bash
lmroot="[PATH TO LM CHECKPOINTS]"
vocabfile="1b_word_vocab.txt"

dataset=$1
export CUDA_VISIBLE_DEVICES=$2

# Iterate over different checkpoints
for pplx in  "59.40738149815701" "66.13396308561414"  "84.06641016142756" "96.56411253523957" "135.9579999482846"
do
	echo "doneat `date`"
	reference="[PATH TO LIBRISPEECH]/librivox-"${dataset}".csv"
	beamdump="[PATH TO BEAMDUMP]"
	python process_beam_dump.py --beam_dump=$beamdump  --beam_dump_with_lm=$beamdump${pplx}.csv  --model=${lmroot}${pplx}/model.pt --vocab=$vocabfile  --reference=$reference
	echo "doneat `date`"
done
