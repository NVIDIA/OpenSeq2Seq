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
     - SacreBLEU(cased)
     - Config file
     - Checkpoint
   * - :doc:`Transformer-big </machine-translation/transformer>`
     - 28.0
     - `transformer-nvgrad.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/transformer-nvgrad.py>`_
     - `link <https://drive.google.com/a/nvidia.com/file/d/1cvR_eCpOMbHdT32dsCveKiCUPvPh4bHC/view?usp=sharing>`_
   * - :doc:`Transformer </machine-translation/transformer>`
     - 26.4
     - `transformer-base.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/transformer-base.py>`_
     - `link <https://drive.google.com/a/nvidia.com/file/d/1wGCZ6ktnW_m9Ie2ynbZ49t332enjbLde/view?usp=sharing>`_
   * - :doc:`ConvS2S </machine-translation/convs2s>`
     - 25.0
     - `en-de-convs2s-8-gpu.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/en-de-convs2s-8-gpu.py>`_
     - `link <https://drive.google.com/a/nvidia.com/file/d/1Xkg5N_nJOvkDx7IDjIAMUWWn3caHNBtj/view?usp=sharing>`_
   * - :doc:`GNMT </machine-translation/gnmt>`
     - 23.0
     - `en-de-gnmt-like-4GPUs.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/en-de-gnmt-like-4GPUs.py>`_
     - TBD

These models have been trained with BPE vocabulary used for text tokenization, available in `wmt16.tar.gz <https://drive.google.com/open?id=1ooQiWhmzmYsk2qMOfaunjTlx_z6lcUyO>`_ . Note that to use pretrained model you will need the same vocabulary which was used for training.  
The model and training parameters can be found in the corresponding config file. We measure BLEU scores using SacreBLEU on detokenized output (cased). 


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

This script will download English-German training data from WMT, clean it, and tokenize using `Google's Sentencepiece library <https://github.com/google/sentencepiece>`_ . By default, the vocabulary size we use is 32,768 for both English and German. 

You can also download the pre-processed dataset which we used for training: `wmt16.tar.gz <https://drive.google.com/open?id=1ooQiWhmzmYsk2qMOfaunjTlx_z6lcUyO>`_ .

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

************************
Using pretrained models
************************

All models have been trained with specific version of tokenizer. So first step would be copy `m_common.model <https://drive.google.com/open?id=1HfBaF_Uk8aGiPWeMIaBRpE5KmC8ryIpk>`_ and `m_common.vocab <https://drive.google.com/open?id=11C4-f2jr2hExIs0QT9sKwUrJedLDml6O>`_ to current folder. 

To translate your English text ``source_txt`` to German you should 

1.tokenize ``source.txt`` into ``source.tok``::

  python tokenizer_wrapper.py --mode=encode --model_prefix=m_common  --text_input=source.txt --tokenized_output=source.tok --vocab_size=32768

2. modify model ``config.py``::


	base_params = {
	  "use_horovod": False,
	  "num_gpus": 1, 
          ...
	  "logdir": "checkpoint/model",
	}
	...
	infer_params = {
	  "batch_size_per_gpu": 256,
	  "data_layer": ParallelTextDataLayer,
	  "data_layer_params": {
	    "src_vocab_file": "m_common.vocab",
	    "tgt_vocab_file": "m_common.vocab",
	    "source_file": "source.tok",
	    "target_file": "source.tok", # this line will be ignored
	    "delimiter":   " ",
	    "shuffle":     False,
	    "repeat":      False,
	    "max_length":  1024,
	  },
	}
        ...
   
2.translate ``source.tok`` into ``output.tok``::
  
  python run.py --config_file=config.py --mode=infer --logdir=checkpoint/model  --infer_output_file=output.tok --num_gpus=1

3.detokenize ``output.tok``::

  python tokenizer_wrapper.py --mode=detokenize --model_prefix=m_common --text_input=output.tok --decoded_output=output.txt 





 
 


