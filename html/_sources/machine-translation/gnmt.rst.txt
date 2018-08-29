.. _gnmt:

GNMT
====

Model
~~~~~
We have 2 models based on RNNs:
  * small NMT (config `en-de-nmt-small.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/18.09/example_configs/text2text/en-de/en-de-nmt-small.py>`_ ) model:

    - the embedding size for source and target is 512
    - 2 birectional LSTM layers in encoder, and 2 LSTM layers in decoder  with state 512
    - the attention mechanism with size 512
  * GNMT-like model based on `Google NMT <https://ai.google/research/pubs/pub45610>`_  (config `en-de-gnmt-like-4GPUs.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/18.09/example_configs/text2text/en-de/en-de-gnmt-like-4GPUs.py>`_ ):

    - the embedding size for source and target is 1024
    - 8 LSTM layers in encoder, and 8 LSTM layers in decoder with state 1024
    - residual connections in encoders and decoders
    - first layer of encoder is bi-directional
    - GNMTv2 attention mechanism
    - the attention layer size 1024


Training
~~~~~~~~~
Both models have been trained with Adam. The small model has following training parameters:
  * intial learning rate to 0.001 
  * Layer-wise Adaptive Rate Clipping (LARC) for gradient clipping.
  * dropout 0.2 

The large model was trained with following parameters:
  * learning rate starting from 0.0008 with staircase decay 0.5 (aka Luong10 scheme)
  * dropout 0.2 

Mixed Precision
~~~~~~~~~~~~~~~
GNMT-like model convergense in float32 and Mixed Precision is almost exactly the same.

.. image:: gnmt-mp.png
