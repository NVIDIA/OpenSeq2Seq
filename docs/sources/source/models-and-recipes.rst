.. _models_and_recipes:

Models and recipes
==================

.. This section will contain information about different models that OpenSeq2Seq
.. supports, exact config parameters to train them, final training/validation/test
.. metrics and links to checkpoints (tensorboards also?) of trained models.

.. note::
    This section is work in progress. Some of the info is not ready. Check back soon.

Currently OpenSeq2Seq has implemented models for machine translation and
automatic speech recognition. Namely, the following models are supported
(to run evaluation or inference the same config can be used with corresponding
mode):

Machine translation
-------------------

Small NMT Model
~~~~~~~~~~~~~~~

Command to train on 1 GPU::

    python run.py --config_file=example_configs/text2text/en-de-nmt-small.py --mode=train_eval

Final metrics: test BLEU score = 20.17 on newstest2014.tok.de using ``multi-bleu.perl`` script from Mosses.
Model checkpoint: `link <https://drive.google.com/file/d/1Lr3eRC4Z3N_FpYzrKtS9809ttBjPJYgT/view?usp=sharing>`_  .

GNMT
~~~~

Model description: https://arxiv.org/abs/1609.08144.

Command to train on 4 GPUs::

    python run.py --config_file=example_configs/text2text/xxx-4GPUs.py --mode=train_eval

Final metrics: test BLEU score = xx.xx. Model checkpoint: link.



Transformers
~~~~~~~~~~~~

Model description: https://arxiv.org/abs/1706.03762.

Command to train on 4 GPUs::

    python run.py --config_file=example_configs/text2text/xxx-4GPUs.py --mode=train_eval

Final metrics: test BLEU score = xx.xx. Model checkpoint: link.

Command to train on 1 GPU::

    python run.py --config_file=example_configs/text2text/xxx-1GPU.py --mode=train_eval

Final metrics: test BLEU score = xx.xx. Model checkpoint: link.

Speech recognition
------------------

Deep Speech 2 based models
~~~~~~~~~~~~~~~~~~~~~~~~~~
Deep Speech 2 model description: https://arxiv.org/abs/1512.02595.

Small model
~~~~~~~~~~~~~

This small Deep Speech 2 like model can be trained on a 'clean' subset of
LibriSpeech in less than a day using a single GPU.

Command to train on 1 GPU::

    python run.py --config_file=example_configs/speech2text/ds2_small_1gpu.py --mode=train_eval --enable_logs

Final metrics: on LibriSpeech dev clean after 12 epochs WER = 9.32% with beam width = 2048.
With beam width = 512, WER = 11.77%.

Model checkpoint: coming soon.


Medium model
~~~~~~~~~~~~~~~~~~~~

This is a medium version of Deep Speech 2 with 3 convolutional and 3 unidirectional GRU layers.

Command to train on 4 GPUs::

    python run.py --config_file=example_configs/speech2text/ds2_medium_4gpus.py --mode=train_eval --enable_logs

Final metrics: on LibriSpeech dev clean after 50 epochs WER = 5.5% with beam width = 2048.
With beam width = 512, WER = 5.96%.

Model checkpoint: coming soon.


Large model
~~~~~~~~~~~~~~~~~~~~

This is a large version of Deep Speech 2 with 2 convolutional and 5 bidirectional GRU layers.

Command to train on 8 GPUs::

    python run.py --config_file=example_configs/speech2text/ds2_large_8gpus.py --mode=train_eval --enable_logs

Final metrics: on LibriSpeech dev clean after 50 epochs WER = 4.59% with beam width = 2048.
With beam width = 512, WER = 4.9%.

Model checkpoint: coming soon.
