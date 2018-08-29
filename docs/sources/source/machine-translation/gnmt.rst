.. _gnmt:

GNMT
====

Model
~~~~~
We have 2 models based on `Google NMT <https://ai.google/research/pubs/pub45610>`_ : 
  * small GNMT (config `en-de-nmt-small.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/en-de-nmt-small.py>`_ ) model:

    - the embedding size for source and target is 512
    - 2 birectional LSTM layers in encoder, and 2 LSTM layers in decoder  with state 512
    - the attention mechanism from the  `Google paper <https://arxiv.org/abs/1609.08144>`_) with size 512
  * large GNMT (config `en-de-gnmt-like-4GPUs.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/en-de-gnmt-like-4GPUs.py>`_ ):

    - the embedding size for source and target is 1024
    - 8 birectional LSTM layers in encoder, and 8 layers in decoder with state 1024 
    - the attention layer size 1024

Both models have been trained on WMT 2014 English-German dataset):  
  * The small model has BLEU 20.23 (`checkpoint  <https://drive.google.com/file/d/1Ty9hiOQx4V28jJmIbj7FWUyw7LVA39SF/view?usp=sharing>`_ ).
  * The large model has BLEU 23.89 (`checkpoint <https://drive.google.com/file/d/1HVc4S8-wv1-AZK1JeWgn6YNITSFAMes_/view?usp=sharing>`_ ).

Training
~~~~~~~~~
Both models have been trained with Adam. The small model has following training parameters:
  * intial learning rate to 0.001 
  * Layer-wise Adaptive Rate Clipping (LARC) for gradient clipping.
  * dropout 0.2 

The large model was trained with following parameters:
  * learning rate starting from 0.0008 with staircase decay 0.5
  * dropout 0.2 

Mixed Precision
~~~~~~~~~~~~~~~
TBD
