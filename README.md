# OpenSeq2Seq: multi-gpu sequence to sequence learning
This is a research project, not an official NVIDIA product.

## Features
1. Sequence to sequence learning
   1. Different cell types: LSTM, GRU, GLSTM, SLSTM
   2. Encoders: RNN-based, unidirectional, bi-directional, GNMT-like
   3. Attention mechanisms: Bahdanau, Luong, GNMT-like
   4. Beam search for inference
2. Data parallel multi-gpu training
3. Distributed (data-parallel) multi-node training using Horovod
4. LARS norm scaling algorithm


## [Documentation](https://github.com/NVIDIA/OpenSeq2Seq/wiki)

* [Getting Started](https://github.com/NVIDIA/OpenSeq2Seq/wiki/Getting-started)
* [Toy Data Example](https://github.com/NVIDIA/OpenSeq2Seq/wiki/Toy-data-example)
* [Training German to English translator](https://github.com/NVIDIA/OpenSeq2Seq/wiki/Training-German-to-English-translator)
* [Models and Recepies](https://github.com/NVIDIA/OpenSeq2Seq/wiki/Models-and-Recepies)
* [Question Answering](https://github.com/NVIDIA/OpenSeq2Seq/blob/master/QuestionAnswering/README.md) (related project)
