:github_url: https://github.com/NVIDIA/OpenSeq2Seq

.. toctree::
   :hidden:
   :maxdepth: 2

   Introduction <self>
   installation
   machine-translation
   speech-recognition
   speech-synthesis
   distr-training
   mixed-precision
   in-depth-tutorials
   interactive-infer-demos
   api-docs/modules


OpenSeq2Seq
===========

OpenSeq2Seq is a TensorFlow-based toolkit for training sequence-to-sequence models:

 * :ref:`machine translation <machine_translation>` (GNMT, Transformer, ConvS2S, ...)
 * :ref:`speech recognition <speech_recognition>` (DeepSpeech2, Wave2Letter, ...)
 * :ref:`speech synthesis <speech_synthesis>` (Tacotron2, ...)

**Main features**:

* Modular design and flexible configuration makes it easy to build new encoder-decoder models 
* efficient mixed-precision training on GPUs
* multi-GPU and multi-node training

To install this toolkit, look at :ref:`installation instructions <installation>`.
Next go to :ref:`in-depth tutorials <in_depth>` section. 
You can also find some useful information in the :ref:`mixed precision  <mixed_precision>` and
:ref:`distributed training <distributed_training>` sections.

**Disclaimer**:
This is a research project, not an official product by NVIDIA.
