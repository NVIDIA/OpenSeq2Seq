Machine Translation
===================

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


#####################################
Creating English-to-German translator
#####################################

Execute the following script to get WMT data::

./get_wmt16_en_dt.sh

This will take a while as a lot of data needs to be downloaded and pre-processed.
After, this is is finished you can try training a "real" model very much like you did above for the toy task::

 python run.py --config_file=example_configs/text2text/en-de-nmt-small.py --mode=train_eval

Before you execute this script, make sure that you've changed ``data_root`` inside ``en-de-nmt-small.py`` to point to the correct WMT data location.
This configuration will take a while to train on a single system. If your GPU does not have enough memory
try reducing the ``batch_size_per_gpu`` parameter. Also, you might want to disable parallel evaluation by using ``--mode=train``.
You can adjusted ``num_gpus`` parameter to train on more than one GPU if available.

*************
Run inference
*************

Once training is done, you can run inference::

    python run.py --config_file=example_configs/text2text/en-de-nmt-small.py --mode=infer --infer_output_file=file_with_BPE_segmentation.txt --num_gpus=1

Note that because BPE-based vocabularies were used during training, the results will contain BPE segmentation.
Also, make sure you use only 1 GPU for inference (``-num_gpus=1``) because otherwise the order of lines in output file is not defined.

*************************
Cleaning BPE segmentation
*************************

Before computing BLEU scores you need to remove BPE segmentation::

  cat file_with_BPE_segmentation.txt | sed -r 's/(@@ )|(@@ ?$)//g' > cleaned_file.txt

*********************
Computing BLEU scores
*********************

Run ```multi-blue.perl``` script on cleaned data::

  ./multi-bleu.perl newstest2014.tok.de < cleaned_file.txt

You should get a BLEU score above 20 for this small model on newstest2014.tok.de.

