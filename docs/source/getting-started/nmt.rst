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

  python run.py --config_file=example_configs/text2text/nmt-reversal-RR.py --mode=infer --infer_output_file=output.txt

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

#. ``example_configs/nmt_reversal-TR.py`` - a model which uses Transformer's encoder and RNN decoder with attention
#. ``example_configs/nmt_reversal-RT.py`` - a model which uses RNN-based encoder Transformer-based decoder


#####################################
Creating English-to-German translator
#####################################

Execute the following script to get WMT data::

./get_wmt16_en_dt.sh

This will take a while as a lot of data needs to be downloaded and pre-processed.
After, this is is finished you can try training a "real" model very much like you did above for the toy task::

 python run.py --config_file=example_configs/text2text/en-de-gnmt-like-4GPUs.py --mode=train_eval

Before you exectute this script, make sure that you've changed ``data_root`` inside ``en-de-gnmt-like-4GPUs.py`` to point to the correct WMT data location.
This configuration will take a while to train on a 4-GPU system. If your GPUs have 16GB memory or more you should be OK. If not,
try reducing the ``batch_size_per_gpu`` parameter. Also, you might want to disable parallel evaluation by using ``--mode=train``.
Make sure you've adjusted ``num_gpus`` parameter to the correct number of GPUs on your system.

*************
Run inference
*************
Once training is done, you can run inference::

    python run.py --config_file=example_configs/text2text/en-de-gnmt-like-4GPUs.py --mode=infer --infer_output_file=file_with_BPE_segmentation.txt
Note that because BPE-based vocabularies were used during training, the results will contain BPE segmentation.

*************************
Cleaning BPE segmentation
*************************
Before computing BLEU scores you need to remove BPE segmentation::

  cat {file_with_BPE_segmentation.txt} | sed -r 's/(@@ )|(@@ ?$)//g' > {cleaned_file.txt}

*********************
Computing BLEU scores
*********************
Run ```multi-blue.perl``` script on cleaned data::

  ./multi-bleu.perl cleaned_newstest2015.tok.bpe.32000.en < cleaned_file.txt

