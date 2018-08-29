.. _interactive-infer-demos:

Interactive Infer Mode
======================

Introduction
------------
Interactive infer is a mode that makes it easy to demo trained models. An
example
`jupyter notebook <https://github.com/NVIDIA/OpenSeq2Seq/blob/18.09-dev/Interactive_Infer_example.ipynb>`_
is provided to showcase how the mode works.

The mode is completely abstracted away from the model. The model is simply run
in infer mode. Data layers however need to be updated to support interactive
infer. Currently, the Text2Text, Text2Speech, and Speech2Text all support
interactive infer.

Please note that the current Text2Text data layer accepts tokenized text, not
raw text. As a result, the input to get_interactive_infer_results() should be
tokenized.

How to run the Jupyter Notebook example
---------------------------------------
The example notebook takes an English sentence as input, produces English audio
via a Text2Speech model, and recognizes the generated speech via a Speech2Text
model. The model requires a trained Text2Speech and a trained Speech2Text model.

Setup for the notebook:
  1. Make a new directory called Infer_S2T in the same directory as the notebook
  2. Copy the Speech2Text configuration file to Infer_S2T and rename it to
     config.py
  3. Copy the Speech2Text model checkpoint to Infer_S2T
  4. Make a new directory called Infer_T2S in the same directory as the notebook
  5. Copy the Text2Speech configuration file to Infer_T2S and rename it to
     config.py
  6. Copy the Text2Speech model checkpoint to Infer_T2S
  7. Run jupyter notebook
  8. Run all cells
  9. Once the input box appears, enter an example sentence surrounded in quotes
     such as ``"Anyone can edit this and generate speech!"``

How to enable interactive infer for a data layer
------------------------------------------------
In order to enable interactive infer for a data layer, a data layer must
additionally implement two functions: create_interactive_placeholders() and
create_feed_dict(). 

In the current framework, model expects all inputs to be
passed inside self._input_tensors['source_tensors']. For
create_interactive_placeholders(), It will suffice to create a
placeholder for each tensor inside this dictionary that is being passed to the
model at infer time from the build_graph() function. Be sure that specify the
placeholder shape and dtype as necessary.

The create_feed_dict() function should take in as input some abstract data
format. It should first check to make sure that the input is formatted correctly.
It will then preprocess the input and create a feed dict that fills in all the
placeholders defined in the create_interactive_placeholders() function.
