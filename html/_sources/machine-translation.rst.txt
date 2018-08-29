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
   * - :doc:`Transformer </machine-translation/transformer>`
     - 26.4
     - `transformer-base.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/18.09/example_configs/text2text/en-de/transformer-base.py>`_     
     - TBD
   * - :doc:`ConvS2S </machine-translation/convs2s>`
     - 25.0
     - `en-de-convs2s-8-gpu.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/18.09/example_configs/text2text/en-de/en-de-convs2s-8-gpu.py>`_
     - TBD
   * - :doc:`GNMT </machine-translation/gnmt>`
     - 23.0
     - `en-de-gnmt-like-4GPUs.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/18.09/example_configs/text2text/en-de/en-de-gnmt-like-4GPUs.py>`_
     - TBD

The model specification and training parameters can be found in the corresponding config file. We measure BLEU scores using SacreBLEU.


.. toctree::
   :hidden:
   :maxdepth: 1

   machine-translation/transformer
   machine-translation/convs2s
   machine-translation/gnmt

################
Getting started 
################

For a simplest example using toy-data (string reversal task) please refer to :doc:`toy models </machine-translation/get_started_nmt>`.

Next let's build a small English-German translation model. This model should train in a reasonable time on a single GPU.

********
Get data
********

Download (this will take some time)::

 scripts/get_en_de.sh



This script will download English-German training data from WMT, clean it, and tokenize using `Google's Sentencepiece library <https://github.com/google/sentencepiece>`_
By default, the vocabluary size we use is 32,768 for both English and German.

********
Training
********

To train a small English-German model:

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


Note that the model output is tokenized. In our case it will output BPE segments instead of words. Therefore, the next step
is to de-tokenize::

 python tokenizer_wrapper.py --mode=detokenize --model_prefix=.../Data/wmt16_de_en/m_common --decoded_output=result.txt --text_input=raw.txt

*********************
Computing BLEU scores
*********************
We measure BLEU scores using SacreBLEU package: (`A Call for Clarity in Reporting BLEU Scores <https://arxiv.org/abs/1804.08771>`_)
Run SacreBleu on detokenized data::

  cat result.txt | sacrebleu -t wmt14 -l en-de > result.txt.BLEU


