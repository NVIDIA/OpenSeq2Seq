# OpenSeq2Seq: sequence to sequence learning
This is a research project, not an official NVIDIA product.

## Features
1. Sequence to sequence learning
   1. Different cell types: LSTM, GRU, GLSTM, SLSTM
   2. Encoders: RNN-based, unidirectional, bi-directional, GNMT-like
   3. Attention mechanisms: Bahdanau, Luong, GNMT-like
   4. Beam search for inference
2. Single box data parallel multi-gpu training
3. Distributed (data-parallel) multi-node, mult-gpu training using Horovod
4. LARS norm scaling algorithm


## [Documentation](https://github.com/NVIDIA/OpenSeq2Seq/wiki)

* [Getting Started](https://github.com/NVIDIA/OpenSeq2Seq/wiki/Getting-started)
* [Toy Data Example](https://github.com/NVIDIA/OpenSeq2Seq/wiki/Toy-data-example)
* [Training German to English translator](https://github.com/NVIDIA/OpenSeq2Seq/wiki/Training-German-to-English-translator)
* [Models and Recepies](https://github.com/NVIDIA/OpenSeq2Seq/wiki/Models-and-Recepies)
* [Distributed training using Horovod](https://github.com/NVIDIA/OpenSeq2Seq/wiki/Distributed-training)
* [Question Answering](https://github.com/NVIDIA/OpenSeq2Seq/blob/master/QuestionAnswering/README.md) (related project)

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
