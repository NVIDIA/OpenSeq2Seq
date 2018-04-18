.. toctree::
   :hidden:
   :maxdepth: 2

   Introduction <self>
   installation-instructions
   getting-started
   models-and-recipes
   distr-training
   mixed-precision
   using-existing-models
   internal-structure
   extending
   api-docs/modules


OpenSeq2Seq
===========

Welcome to OpenSeq2Seq toolkit! 

The goal of this project is to greatly simplify working with sequence-to-sequence models in TensorFlow.
OpenSeq2Seq has the following features:

* Supports multi-GPU and multi-node training with just 1 line change in the config

* A built-in support for GPU mixed-precision training makes the models
  run faster and use less memory

* Modular design and flexible configuration makes it easy to build new models and 
  experiment with different encoder-decoder
  combinations, e.g. you can combine CNN-based encoder with RNN-based decoder

* Supports different input-output modalities, i.e. speech-to-text, text-to-text etc

* When training the model the toolkit will automatically:

   * print different training and validation metrics
   * log gradient summaries to make it easier to debug if something goes wrong
   * saves the best models based on the validation loss, so you will never lose
     a good checkpoint
   * saves all outputs, configs and git statistics to make it possible to
     completely reproduce all experiments


To start using this toolkit, look at the
:ref:`installation instructions <installation-instructions>` and then
see the :ref:`getting started <getting-started>` page.
If you are already familiar with the basics and have
everything set up, you can find some useful information in the
:ref:`adding new models <extending>` section or in the :ref:`API documentation <api-docs>`.
