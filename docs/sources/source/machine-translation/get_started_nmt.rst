.. _get_started_nmt:

Getting Started
===============

##############################
Toy task - reversing sequences
##############################

You can tests how things work on the following end-to-end toy task.
First, execute::

./create_toy_data

This should create ``toy_text_data`` folder on disk. This is a data for the toy
machine translation problem where the task is to learn to reverse sequences.

For example, if src=``α α ζ ε ε κ δ ε κ α ζ`` then, "correct" translation is tgt=``ζ α κ ε δ κ ε ε ζ α α``.

To train a simple, RNN-based encoder-decoder model with attention, execute the following command::

 python run.py --config_file=example_configs/text2text/nmt-reversal-RR.py --mode=train_eval

This will train a model and perform evaluation on the "dev" dataset in parallel.
To view the progress of training, start Tensorboard::

  tensorboard --logdir=.

To run "inference" mode on the "test" execute the following command::

  python run.py --config_file=example_configs/text2text/nmt-reversal-RR.py --mode=infer --infer_output_file=output.txt --num_gpus=1

Once, finished, you will get inference results in ``output.txt`` file. You can measure how
well it did by launching Mosses's script::

 ./multi-bleu.perl toy_text_data/test/target.txt < output.txt

You should get above 0.9 (which corresponds to BLEU score of 90).
To train a "Transformer"-based model (see `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ paper) use ``example_configs/nmt_reversal-TT.py``
configuration file.

********************
Feeling adventurous?
********************

One of the main goals of OpenSeq2Seq is to allow you easily experiment with different architectures. Try out these configurations:

#. ``example_configs/nmt_reversal-CR.py`` - a model which uses Convolutional encoder and RNN decoder with attention
#. ``example_configs/nmt_reversal-RC.py`` - a model which uses RNN-based encoder and Convolutional decoder
#. ``example_configs/nmt_reversal-TT.py`` - a model which uses Transformer-based encoder and Transformer-based decoder


