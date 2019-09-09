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

## Building

Please see the detailed instructions in [OpenSeq2Seq documentation](https://nvidia.github.io/OpenSeq2Seq/html/installation.html#how-to-build-a-custom-native-tf-op-for-ctc-decoder-with-language-model-optional).

