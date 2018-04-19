# DeepSpeech custom decoder

This folder contains a CTC beam search decoder that scores beams using a KenLM-based N-gram language model. 

## Installation


## Required Dependencies

Running inference might require some runtime dependencies to be already installed on your system. Those should be the same, whatever the bindings you are using:
* libsox2
* libstdc++6
* libgomp1
* libpthread


## Build Requirements

You'll need the following pre-requisites downloaded/installed:

* [TensorFlow source and requirements](https://www.tensorflow.org/install/install_sources)
* [libsox](https://sourceforge.net/projects/sox/)


## Preparation

Create a symbolic link in your TensorFlow checkout to `ctc_decoder_with_lm` directory. If your DeepSpeech and TensorFlow checkouts are side by side in the same directory, do:

```
cd tensorflow
ln -s ../OpenSeq2Seq/ctc_decoder_with_lm ./
```

## Building

## Step 1 : Build Tensorflow
You need to re-build TensorFlow.
Follow the [instructions](https://www.tensorflow.org/install/install_sources) on the TensorFlow site for your platform, up to the end of 'Configure the installation':

```
./configure
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package 
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install /tmp/tensorflow_pkg/tensorflow*.whl
sudo pip install --upgrade numpy
```

## Step 2: Build CTC beam search decoder:

```
bazel build -c opt --copt=-O3 --config=cuda //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //ctc_decoder_with_lm:libctc_decoder_with_kenlm.so //ctc_decoder_with_lm:generate_trie 
cp bazel-bin/ctc_decoder_with_lm/*.so OpenSeq2Seq/ctc_decoder_with_lm/
cp bazel-bin/ctc_decoder_with_lm/generate_trie OpenSeq2Seq/ctc_decoder_with_lm/
```
