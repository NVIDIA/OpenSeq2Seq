## Download the data:
```
$ get_wmt16_en_de.sh
```

Edit the 'example_configs/nmt_one.json' file and replace [WMT16_DATA_LOCATION] with wmt16_en_dt folder full name.

## Run training
Edit "num_gpus" section of nmt.json - set it to the number of GPUs you want to use.
```
python run.py --config_file=example_configs/nmt_one.json --logdir=nmt_one --checkpoint_frequency=2000 --summary_frequency=50 --eval_frequency=1000
```
* If you are getting OOM exceptions try decreasing batch_size parameter in ```nmt_one.json```

## Run Inference
```
python run.py --config_file=example_configs/nmt_one.json --logdir=nmt_one --mode=infer --inference_out=wmt_pred.txt
```

Before we calculate the BLEU score, we must remove the BPE segmentations from the translated outputs.

Clean ```wmt_pred.txt``` and ```newstest2015.tok.bpe.32000.en``` from BPE segmentation

## Cleaning BPE segmentation
```
$ cat {file_with_BPE_segmentation} | sed -r 's/(@@ )|(@@ ?$)//g' > {cleaned_file}
```

Run ```multi-blue.perl``` script on cleaned data.
```
$ ./multi-bleu.perl cleaned_newstest2015.tok.bpe.32000.en < cleaned_wmt_pred.txt
```