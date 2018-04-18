# OpenSeq2Seq: toolkit for distributed and mixed precision training of sequence-to-sequence models
This is a research project, not an official NVIDIA product.

OpenSeq2Seq main goal is to allow researchers to most effectively
explore various
sequence-to-sequence models. The
efficiency is achieved by fully supporting
distributed and mixed-precision training.
OpenSeq2Seq is built using Tensorflow and provides all the necessary
building blocks for training encoder-decoder
models for neural machine translation
and automatic speech recognition.
We plan to extend it with other modalities
in the future.

## Features
1. Sequence to sequence learning
   1. Neural Machine Translation
   2. Automatic Speech Recognition
2. Data-parallel distributed training
   1. Multi-GPU
   2. Multi-node
3. Mixed precision training for NVIDIA Volta GPUs


## Documentation
https://nvidia.github.io/OpenSeq2Seq/

## Acknowledgments
Speech-to-text workflow uses some parts of [Mozilla DeepSpeech](https://github.com/Mozilla/DeepSpeech) project.

Text-to-text workflow uses some functions from [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) and [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt).

## Related resources
* [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
* [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt)
* [OpenNMT](http://opennmt.net/)
* [Sockeye](https://github.com/awslabs/sockeye)
* [TF-seq2seq](https://github.com/google/seq2seq)
* [Moses](http://www.statmt.org/moses/)

