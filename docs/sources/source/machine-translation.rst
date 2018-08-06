.. _machine_translation:

Machine Translation
===================

######
Models
######

Currently we support following models:

.. list-table::
   :widths: 1 1 2 1 
   :header-rows: 1

   * - Model description
     - BLEU
     - Config file
     - Checkpoint
   * - :doc:`GNMT </machine-translation/gnmt>`
     - 23.89
     - `en-de-gnmt-like-4GPUs.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/en-de-gnmt-like-4GPUs.py>`_     
     - `link <https://drive.google.com/file/d/1HVc4S8-wv1-AZK1JeWgn6YNITSFAMes_/view?usp=sharing>`_
   * - :doc:`Transformer </machine-translation/transformer>`
     - 26.17
     - `transformer-big.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/transformer-big.py>`_     
     - `link <https://drive.google.com/file/d/151R6iCCtehRLpnH3nBmhEi_nhNO2mXW8/view?usp=sharing>`_
   * - :doc:`ConvS2S </machine-translation/convs2s>`
     - 25.40
     - `en-de-convs2s.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/en-de-convs2s.py>`_
     - link

The model specification and training parameters can be found in the corresponding config file.
The BLEU score for these models was measured on newstest2014.tok.de file with ``multi-bleu.perl`` script from Mosses). 

.. toctree::
   :hidden:
   :maxdepth: 1

   machine-translation/gnmt
   machine-translation/transformer
   machine-translation/convs2s

################
Getting started 
################

You can start with these :doc:`toy models </machine-translation/get_started_nmt>`.
Next let's build a small English-German translation engine based on Google NMT model.

********
Get data
******** 

Download (this will take some time) and preprocess WMT16 English-German dataset::

./get_wmt16_en_dt.sh


********
Training
********

Next let's train a small English-German model. For this: 

* change ``data_root`` inside ``en-de-nmt-small.py`` to the WMT data location 
* adjust ``num_gpus`` to train on more than one GPU (if available).

Start training::

 python run.py --config_file=example_configs/text2text/en-de-nmt-small.py --mode=train_eval

If your GPU does not have enough memory, reduce the ``batch_size_per_gpu``. Also, you might want to disable parallel evaluation by using ``--mode=train``. 

*************
Inference
*************

Once training is done (this can take a while on a single GPU), you can run inference::

    python run.py --config_file=example_configs/text2text/en-de-nmt-small.py --mode=infer --infer_output_file=file_with_BPE_segmentation.txt --num_gpus=1

Note that because BPE-based vocabularies were used during training, the results will contain BPE segmentation.
Also, make sure you use only 1 GPU for inference (``-num_gpus=1``) because otherwise the order of lines in output file is not defined.

*********************
Computing BLEU scores
*********************

Before computing BLEU scores you need to remove BPE segmentation::

  cat file_with_BPE_segmentation.txt | sed -r 's/(@@ )|(@@ ?$)//g' > cleaned_file.txt

Then run ```multi-blue.perl``` script on cleaned data::

  ./multi-bleu.perl newstest2014.tok.de < cleaned_file.txt

You should get a BLEU score above 20 for GNMT-small model on newstest2014.tok.de.

