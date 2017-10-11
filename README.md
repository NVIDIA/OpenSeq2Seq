# OpenSeq2Seq: multi-gpu sequence to sequence learning
This is a research project, not an official NVIDIA product.

## Features
1. Sequence to sequence learning
   1. Different cell types: LSTM, GRU, GLSTM
   2. Attention mechanisms: Bahdanau and Luong
   3. Beam search for inference
2. Data parallel multi-gpu training
3. LARS norm scaling algorithm

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
Full documentation is available on the [repository wiki.](https://github.com/NVIDIA/OpenSeq2Seq/wiki)
