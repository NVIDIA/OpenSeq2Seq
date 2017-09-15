# OpenSeq2Seq: multi-gpu sequence to sequence learning
This is a research project, not an official NVIDIA product.

# Getting started

## Requirements
* Python 3.6
* Tensorflow r1.2 or r1.3 (with GPU support)
* NLTK v3.2.3+

## Unit tests
Checkout the code and make sure the following test pass:
You should see **OK** after each test
```
./create_toy_data.sh
python -m unittest test/data_layer_tests.py
python -m unittest test/model_tests.py
```

## Example run on toy data
When you execute ```./create_toy_data.sh``` script, it will create the following toy data:

* Source: random sequence of numbers

* Target: reversed source sequence

The task is to learn to reverse randomly generated sequences. You can test single or multi-gpu training.

**Single GPU:**
```
$ python run.py --config=example_configs/toy_data_config.json --mode=train --logdir=ModelAndLogFolder
```
* You can monitor training progress with Tensorboard: ```tensorboard --logdir=ModelAndLogFolder```


**Multi-GPU:**

We follow *data parallel* approach for training. Each GPU receives a full copy of the model and its own mini-batch of data.

To start multi-GPU training, set ```"num_gpus"``` parameter in the model config accordingly.

The ```"batch_size"``` parameter specifies batch size *per GPU*. Hence, global (or algorithmic) batch size will be equal to ```"num_gpus" * "batch_size"```

*example_configs/toy_data_config_2GPUs.json* shows how to use the multi-gpu case

```
$ python run.py --config=example_configs/toy_data_config_2GPUs.json --mode=train --logdir=ModelAndLogFolder
```

This will begin the training process and save the checkpoints to the ModelAndLogFolder directory

## Example inference
Once the model has been trained and saved in ``ModelAndLogFolder``, run:
```
$ python run.py --config=example_configs/toy_data_config_2GPUs.json --mode=infer --logdir=ModelAndLogFolder --inference_out=pred.txt
```

## BLEU Score Calculation using Moses Script
```
$ ./multi-bleu.perl test/toy_data/test/target.txt < pred.txt
```
If you just used the provided configs, your BLUE score should be > 98 for both single and 2 gpu runs of the toy task.

# Training German to English translation

## First, get the data:
Download and execute [this script](https://github.com/google/seq2seq/blob/master/bin/data/wmt16_en_de.sh)

Edit the 'nmt.json' file and replace [WMT16_DATA_LOCATION] with correct data location.

## Run training
Edit "num_gpus" section of nmt.json - set it to the number of GPUs you want to use.
```
python run.py --config_file=example_configs/nmt.json --logdir=nmt --checkpoint_frequency=2000 --summary_frequency=50 --eval_frequency=1000
```
* If you are getting OOM exceptions try decreasing batch_size parameter in ```nmt.json```

## Run Inference
```
python run.py --config_file=example_configs/nmt.json --logdir=nmt --mode=infer --inference_out=wmt_pred.txt
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

## Models and Recepies
| MODEL          | Language Pair | Steps  | Elapsed time | newstest2015 BLUE | Pre-trained model |
|:--------------:|:--:|:-------------:|:---: |:---: |:---: |
| nmt.json       | De->En |    59,208    |  21h 7m on 1x Quadro GP100 (16GB) | 23.36    | |
| nmt_4gpu.json  | De->En |    51,100    | - |  25.21   | [Download](https://drive.google.com/file/d/0B8NYpqxq57n7YUhHRlY4Y09ud00/view?usp=sharing)|


For this *example* config you should get BLEU score around 22.66.

On a single Quadro GP100 (16GB) it takes around 24 hours (4 epochs) to get this result.

Train longer (and/or using more GPUs) to get better results.

### Authors
[Oleksii Kuchaiev](https://github.com/okuchaiev) and [Siddharth Bhatnagar](https://github.com/siddharthbhatnagar) (internship work at NVIDIA)

Contributions are welcome!

## Related resources
* [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt)
* [OpenNMT (Torch)](http://opennmt.net/)
* [OpenNMT (Pytorch)](https://github.com/OpenNMT/OpenNMT-py)
* [Tf-seq2seq](https://github.com/google/seq2seq)
* [Moses](http://www.statmt.org/moses/)

## References
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/abs/1703.03906)
* [Effective approaches to attention-based neural machine translation](https://arxiv.org/abs/1508.04025)
