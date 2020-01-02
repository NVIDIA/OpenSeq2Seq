[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/documentation-github.io-blue.svg)](https://nvidia.github.io/OpenSeq2Seq/html/index.html)
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

Beam search decoder with language model re-scoring implementation (in `decoders`) is based on [Baidu DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech).

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
@misc{openseq2seq,
    title={Mixed-Precision Training for NLP and Speech Recognition with OpenSeq2Seq},
    author={Oleksii Kuchaiev and Boris Ginsburg and Igor Gitman and Vitaly Lavrukhin and Jason Li and Huyen Nguyen and Carl Case and Paulius Micikevicius},
    year={2018},
    eprint={1805.10387},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Install Decoders

### Install boost/automake and bison
```
sudo apt-get install libboost-all-dev -y
sudo apt-get install automake -y
sudo apt-get install bison -y
```

### Install SWIG
 ```
 git clone https://github.com/swig/swig.git
cd swig
./autogen.sh
./configure
make
sudo make install
 ```
 #### Test once
 ```
 $ swig
 ```
 if you encounter 
```
$ swig: error while loading shared libraries: libpcre.so.1: cannot open shared object file: No such file or directory
```
## Install PCRE
```
cd /usr/local/src
sudo curl --remote-name ftp://ftp.csx.cam.ac.uk/pub/software/programming/pcre/pcre-8.42.tar.gz

tar -xzvf pcre-8.42.tar.gz
cd pcre-8.42
sudo ./configure --prefix=/usr/local/mac-dev-env/pcre-8.42
sudo make
sudo make install 
sudo ln -s mac-dev-env/pcre-8.42 /usr/local/pcre
echo 'export PATH=/usr/local/pcre/bin:$PATH' >> ~/.bash_profile
source ~/.bash_profile
cd .libs
sudo mv -v libpcre.so.* /usr/lib/
```
If the above doesnt works then please use the latest version as follows:

```
sudo curl --remote-name https://ftp.pcre.org/pub/pcre/pcre-8.43.tar.bz2
tar xjf  pcre-8.43.tar.bz2 
cd pcre-8.43/
sudo ./configure --prefix=/usr/local/mac-dev-env/pcre-8.43
sudo make
sudo make install 
sudo ln -s mac-dev-env/pcre-8.43 /usr/local/pcre
echo 'export PATH=/usr/local/pcre/bin:$PATH' >> ~/.bash_profile
source ~/.bash_profile
cd .libs
sudo mv -v libpcre.so.* /usr/lib/
```

If the symlink is already used..either delete or use another symlink
## Final Output

```
$ swig
Must specify an input file. Use -help for available options.
```

### ThankYou
