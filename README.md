<div align="center">
  <img src="./docs/logo-shadow.png" alt="OpenSeq2Seq" width="250px">
  <br>
</div>

# OpenSeq2Seq: toolkit for distributed and mixed precision training of sequence-to-sequence models

This is a research project, not an official NVIDIA product.

Documentation: https://nvidia.github.io/OpenSeq2Seq/

OpenSeq2Seq main goal is to allow researchers to most effectively
explore various
sequence-to-sequence models. The
efficiency is achieved by fully supporting
distributed and mixed-precision training.
OpenSeq2Seq is built using TensorFlow and provides all the necessary
building blocks for training encoder-decoder
models for neural machine translation
and automatic speech recognition.
We plan to extend it with other modalities
in the future.

## Features
1. Sequence to sequence learning. Currently implemented:
   1. Neural Machine Translation (text2text)
   2. Automatic Speech Recognition (speech2text)
   3. Speech Synthesis (text2speech)
2. Data-parallel distributed training
   1. Multi-GPU
   2. Multi-node
3. Mixed precision training for NVIDIA Volta GPUs


## Software Requirements
1. TensorFlow >= 1.9
2. Horovod >= 0.12.0 (using Horovod is not required, but is highly recommended for multi-GPU setup)

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

## Paper
If you use OpenSeq2Seq, please cite [this paper](https://arxiv.org/abs/1805.10387)
```
@article{openseq2seq,
  title={
OpenSeq2Seq: extensible toolkit for distributed and mixed precision training of sequence-to-sequence models},
  author={Kuchaiev, Oleksii and Ginsburg, Boris and Gitman, Igor and Lavrukhin, Vitaly and  Case, Carl and Micikevicius, Paulius},
  journal={arXiv preprint arXiv:1805.10387},
  year={2018}
}
```
