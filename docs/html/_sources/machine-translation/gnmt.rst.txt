.. _gnmt:

GNMT
====

Model
~~~~~
This model is based on `Google NMT model <https://ai.google/research/pubs/pub45610>`_ (see also `paper <https://arxiv.org/abs/1609.08144>`_). 
We have 2 English-to-German models trained on WMT 2014 English-German dataset:
 
* small GNMT model (config `en-de-nmt-small.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/en-de-nmt-small.py>`_ ),
  which has 2 birectional LSTM layers in encoder and  2 layers in decoder. 
  This model achieve BLEU score 20.23 on WMT 2014 English-to-German translation task 
  ( `checkpoint  <https://drive.google.com/file/d/1Ty9hiOQx4V28jJmIbj7FWUyw7LVA39SF/view?usp=sharing>`_ ). 

* large GNMT (config file `en-de-gnmt-like-4GPUs.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/en-de-gnmt-like-4GPUs.py>`_ )  model which has 8 birectional LSTM layers in encoder and 8 layers in decoder. 
  This model has BLEU score 23.89 on WMT 2014 English-to-German translation task 
  ( `checkpoint <https://drive.google.com/file/d/1HVc4S8-wv1-AZK1JeWgn6YNITSFAMes_/view?usp=sharing>`_ ).

Training
~~~~~~~~~
Both models have been trained using Adam ...


Mixed Precision
~~~~~~~~~~~~~~~
TBD
