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

Quick Deep Speech 2 Model
~~~~~~~~~~~~~~~~~~~~~~~~~

This small Deep Speech 2 like model can be trained on a 'clean' subset of
LibriSpeech in less than a day using a single GPU.

Command to train on 1 GPU::

    python run.py --config_file=example_configs/speech2text/ds2_quick_librispeech_config.py --mode=train_eval

Final metrics: WER = 9.29% on LibriSpeech dev clean with a beam width = 2048 after 12 epochs.
Model checkpoint: `link <https://drive.google.com/file/d/1O3OHwPdPikI812pkF2vRcxlUSQNMLMLN/view?usp=sharing>`_.


Deep Speech 2
~~~~~~~~~~~~~

Model description: https://arxiv.org/abs/1512.02595.

Command to train on 4 GPUs::

    python run.py --config_file=example_configs/speech2text/ds2_librispeech_adam_config.py --mode=train_eval

Final metrics: WER = 6.04% on LibriSpeech dev clean with a beam width = 2048 after 100 epochs.
Model checkpoint: `link https://drive.google.com/file/d/1TBd5VA6EHLBxLzuMZIe3Z0D0EQJiYVPi/view?usp=sharing>`_.
