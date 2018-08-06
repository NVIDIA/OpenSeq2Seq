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
   api-docs/modules


OpenSeq2Seq
===========

OpenSeq2Seq is a TensorFlow toolkit based on the sequence-to-sequence paradigm for

 * :ref:`machine translation <machine_translation>` (GNMT, Transformer, ConvS2S,..)
 * :ref:`speech recogtnition <speech_recognition>` (DeepSpeech2, Wave2Letter,..),
 * :ref:`speech synthesis <speech_synthesis>` (Tacotron2,..), 

**Main features**:

* Modular design and flexible configuration makes it easy to build new encoder-decoder models 
* multi-GPU and multi-node training
* efficient mixed-precision training on GPUs.

To install this toolkit, look at :ref:`installation instructions <installation>`. 
For more detailed tutorials you can look into :ref:`in-depth tutorials <in_depth>` section.

If you are already familiar with the basics and have
everything set up, check out the available :ref:`models and recipes <models_and_recipes>`.
You can also find some useful information in the :ref:`mixed precision training <mixed_precision>` and
:ref:`distributed training <distributed_training>` sections or look through our
:ref:`API documentation <api-docs>`.

**Disclaim**:
This is a research project, not an official product by NVIDIA

