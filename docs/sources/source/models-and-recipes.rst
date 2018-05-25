.. _models_and_recipes:

Models and recipes
==================

.. This section will contain information about different models that OpenSeq2Seq
.. supports, exact config parameters to train them, final training/validation/test
.. metrics and links to checkpoints (tensorboards also?) of trained models.

.. note::
    This section is work in progress. Some of the info is not ready. Check back soon.

Currently OpenSeq2Seq has implemented models for machine translation and
automatic speech recognition. To train models you can use the following
commands (don't forget to substitute valid config_file path there).

With Horovod (highly recommended when using multiple GPUs)::

    mpirun --allow-run-as-root --mca orte_base_help_aggregate 0 -mca btl ^openib -np 4 -H localhost:4 -bind-to none -map-by slot -x LD_LIBRARY_PATH python run.py --config_file=... --mode=train_eval --enable_logs

Without Horovod::

    python run.py --config_file=... --mode=train_eval --enable_logs

The description of implemented models is available in the next sections:

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

Final metrics: test BLEU score = xx.xx.

Model checkpoint: link.


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
Original Deep Speech 2 model description: https://arxiv.org/abs/1512.02595.
The table below contains description and results of
Deep Speech 2 based models available in OpenSeq2Seq.

WER-512 and WER-2048 is word error rate obtained with beam width of 512 and 2048
correspondingly. For more details about model descriptions and training setup,
have a look at the `configuration files <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/>`_.

.. list-table::
   :widths: 1 1 1 1 1 1
   :header-rows: 1

   * - Config file
     - WER-512
     - WER-2048
     - Training setup and additional comments
     - Short description of the model
     - Checkpoint
   * - `ds2_large_8gpus.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/ds2_large_8gpus.py>`_
     - 4.90%
     - 4.59%
     - This model was trained for 50 epochs using SGD with Momentum and LARC on
       the full LibriSpeech in a few days using Horovod on eight GPUs.
     - This model has 2 convolutional layers and 5 bidirectional
       GRU layers with 800 units.
     - `link <https://drive.google.com/file/d/1gfGg3DzXviNhYlIyxl12gWp47R8Uz-Bf/view?usp=sharing>`_
   * - `ds2_medium_4gpus.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/ds2_medium_4gpus.py>`_
     - 5.96%
     - 5.50%
     - This model was trained for 50 epochs using Adam on the full
       LibriSpeech in a few days using Horovod on four GPUs.
     - This model has 3 convolutional layers and 3 unidirectional
       GRU layers with 1024 units.
     - Coming soon.
   * - `ds2_small_1gpu.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/ds2_small_1gpu.py>`_
     - 11.77%
     - 9.32%
     - This model was trained for 12 epochs using Adam on a "clean" subset of
       LibriSpeech in less than a day using a single GPU.
     - This model has 2 convolutional layers and 2 unidirectional
       GRU layers with 512 units.
     - `link <https://drive.google.com/file/d/1-OEvxyg7rCogZhejen7pNuKkgvuwCdbk/view?usp=sharing>`_
