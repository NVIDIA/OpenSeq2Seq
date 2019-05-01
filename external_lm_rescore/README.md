# External LM re-scoring

We use Transformer-XL code taken from: https://github.com/kimiyoung/transformer-xl/tree/master/pytorch

1) Take any pre-trained ASR model, add `'infer_logits_to_pickle': True` to "decoder_params" section of the model's config file and put a required CSV file in `"dataset_files"` field of "infer_params" section.

2) Run inference (in order to dump logits from the model to pickle file):
```
python run.py --mode=infer --config="MODEL_CONFIG" --logdir="MODEL_CHECKPOINT_DIR" --num_gpus=1 --batch_size_per_gpu=1 --decoder_params/use_language_model=False --infer_output_file=model_output.pickle
```

3) Run beam search decoder (with specific ALPHA, BETA and BEAM_WIDTH hyperparameters) and dump beams:
```
python scripts/decode.py --logits=model_output.pickle --labels="CSV_FILE" --lm="LM_BINARY"  --vocab="ALPHABET_FILE" --alpha=ALPHA --beta=BETA --beam_width=BEAM_WIDTH --dump_all_beams_to=BEAMS.txt
```

4) Download checkpoints for pre-trained Transformer-XL LM: [LibriSpeech](https://drive.google.com/a/nvidia.com/file/d/15--z08YNePr8Fgx4cnY4zR37QPZ3ZfMf/view?usp=sharing) and [WSJ](https://drive.google.com/a/nvidia.com/file/d/13D4Hwr_fOd85tkzLxDchEsodlOWdyBhY/view?usp=sharing).  Then have a look at [`run_lm_exp.sh`](https://github.com/NVIDIA/OpenSeq2Seq/blob/master/external_lm_rescore/run_lm_exp.sh) for example on how to run re-scoring.
