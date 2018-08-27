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
     - SacreBLEU
     - Config file
     - Checkpoint
   * - :doc:`GNMT </machine-translation/gnmt>`
     - 23.0
     - `en-de-gnmt-like-4GPUs.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/en-de-gnmt-like-4GPUs.py>`_     
     - TBD
   * - :doc:`Transformer </machine-translation/transformer>`
     - 26.4
     - `transformer-big.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/transformer-big.py>`_     
     - TBD
   * - :doc:`ConvS2S </machine-translation/convs2s>`
     - 25.0
     - `en-de-convs2s.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/en-de-convs2s.py>`_
     - TBD

The model specification and training parameters can be found in the corresponding config file. We measure BLEU scores using SacreBLEU.


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

 scripts/get_en_de.sh


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

    python run.py --config_file=example_configs/text2text/en-de-nmt-small.py --mode=infer --infer_output_file=raw.txt --num_gpus=1

Note that because BPE-based vocabularies were used during training, the results will contain BPE segmentation.
Also, make sure you use only 1 GPU for inference (``-num_gpus=1``) because otherwise the order of lines in output file is not defined.

*********************
Computing BLEU scores
*********************

Before computing BLEU scores you need to detokenize::

  python tokenizer_wrapper.py --mode=detokenize --model_prefix=.../Data/wmt16_de_en/m_common --decoded_output=result.txt --text_input=raw.txt


Then run SacreBleu on detokenized data::

  cat result.txt | sacrebleu -t wmt14 -l en-de > result.txt.BLEU


