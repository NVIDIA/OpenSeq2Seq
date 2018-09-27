<div align="center">
  <img src="./docs/logo-shadow.png" alt="OpenSeq2Seq" width="250px">
  <br>
</div>

# OpenSeq2Seq: toolkit for distributed and mixed precision training of sequence-to-sequence models

OpenSeq2Seq main goal is to allow researchers to most effectively explore various
sequence-to-sequence models. The efficiency is achieved by fully supporting
distributed and mixed-precision training.
OpenSeq2Seq is built using TensorFlow and provides all the necessary
building blocks for training encoder-decoder models for neural machine translation, automatic speech recognition, speech synthesis, and language modeling.

## Documentation and installation instructions 
https://nvidia.github.io/OpenSeq2Seq/

## Features
1. Models for:
   1. Neural Machine Translation
   2. Automatic Speech Recognition
   3. Speech Synthesis
   4. Language Modeling
   5. NLP tasks (sentiment analysis)
2. Data-parallel distributed training
   1. Multi-GPU
   2. Multi-node
3. Mixed precision training for NVIDIA Volta/Turing GPUs

## Software Requirements
1. Python >= 3.5
2. TensorFlow >= 1.10
3. CUDA >= 9.0, cuDNN >= 7.0 
4. Horovod >= 0.13 (using Horovod is not required, but is highly recommended for multi-GPU setup)

## Acknowledgments
Speech-to-text workflow uses some parts of [Mozilla DeepSpeech](https://github.com/Mozilla/DeepSpeech) project.

Text-to-text workflow uses some functions from [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) and [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt).

## Disclaimer
This is a research project, not an official NVIDIA product.

## Related resources
* [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
* [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt)
* [OpenNMT](http://opennmt.net/)
* [Neural Monkey](https://github.com/ufal/neuralmonkey)
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
