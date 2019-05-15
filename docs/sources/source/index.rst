:github_url: https://github.com/NVIDIA/OpenSeq2Seq

.. toctree::
   :hidden:
   :maxdepth: 2

   Introduction <self>
   installation
   using-existing-models
   distr-training
   mixed-precision
   optimizers
   speech-recognition
   speech-commands
   speech-synthesis
   machine-translation
   language-model
   sentiment-analysis
   image-classification
   interactive-infer-demos
   adding-new-models
   api-docs/modules


OpenSeq2Seq
===========

OpenSeq2Seq is a TensorFlow-based toolkit for sequence-to-sequence models:

 * :ref:`machine translation <machine_translation>` (GNMT, Transformer, ConvS2S, ...)
 * :ref:`speech recognition <speech_recognition>` (DeepSpeech2, Wave2Letter, Jasper, ...)
 * :ref:`speech commands <speech_commands>` (RN-50, Jasper)
 * :ref:`speech synthesis <speech_synthesis>` (Tacotron2, WaveNet...)
 * :ref:`language model <language_model>` (LSTM, ...)
 * :ref:`sentiment analysis <sentiment_analysis>` (SST, IMDB, ...)
 * :ref:`image classification <image_classification>` (ResNet-50)

**Main features**:

* modular architecture that allows assembling of new models from available components
* support for mixed-precision training, that utilizes Tensor Cores in NVIDIA Volta/Turing GPUs
* fast Horovod-based distributed training supporting both multi-GPU and multi-node modes

To install this toolkit, look at :ref:`installation instructions <installation>`.
Next go to :ref:`in-depth tutorials <in_depth>` section. 
You can also find some useful information in the :ref:`mixed precision  <mixed_precision>` and
:ref:`distributed training <distributed_training>` sections.

**Reference**:

 1. `Oleksii Kuchaev et al. Mixed-Precision Training for NLP and Speech Recognition with OpenSeq2Seq, 2018 <https://arxiv.org/abs/1805.10387>`_
 2. `Jason Li et al. Jasper: An End-to-End Convolutional Neural Acoustic Model, 2019 <https://arxiv.org/abs/1904.03288>`_

**Disclaimer**:
This is a research project, not an official product by NVIDIA.

