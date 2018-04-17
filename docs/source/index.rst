.. toctree::
   :hidden:
   :maxdepth: 2

   self
   installation-instructions
   getting-started
   tutorials
   extending
   api-docs/modules


OpenSeq2Seq
===========

Welcome to OpenSeq2Seq toolkit! The goal of this project is to greatly simplify
workflow with sequence-to-sequence models in TensorFlow.

OpenSeq2Seq v0.2 has the following features:

* Enables multi-GPU and multi-node training with just 1 line change in the config

* Has a built-in support for mixed-precision (which makes the models
  run faster and use less memory)

* Allows researchers to easily experiment with different encoder-decoder
  combinations, e.g. you can combine CNN-based encoder with RNN-based decoder
  or experiment with Transformers architecture with just a few lines change in
  the config

* Supports different input-output modalities, i.e.
  speech-to-text or text-to-text and different execution modes, i.e. training,
  evaluation and inference

* When training the model the toolkit will automatically:

   * print different training and validation metrics
   * log gradient summaries to make it easier to debug if something goes wrong
   * saves the best models based on the validation loss, so you will never lose
     a good checkpoint
   * saves all outputs, configs and git statistics to make it possible to
     completely reproduce all experiments

* Has modular design and flexible configuration which makes it easy to extend
  OpenSeq2Seq with your custom models

To start using this toolkit, look at the
:ref:`installation instructions <installation-instructions>` and then
see the :ref:`getting started <getting-started>` page.
If you are already familiar with the basics and have
everything set up, you can find some useful information in the
:ref:`adding new models <extending>` section or in the :ref:`API documentation <api-docs>`.