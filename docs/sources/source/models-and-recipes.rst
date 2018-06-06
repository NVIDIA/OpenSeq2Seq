.. _models_and_recipes:

Models and recipes
==================

.. This section will contain information about different models that OpenSeq2Seq
.. supports, exact config parameters to train them, final training/validation/test
.. metrics and links to checkpoints (tensorboards also?) of trained models.

.. note::
    Currently OpenSeq2Seq has model implementations for machine translation and
    automatic speech recognition. All models work both in float32 and mixed precision.
    We recommend you use :ref:`mixed precision training <mixed_precision>` when training on Volta GPUs.


To train models you can use the following
commands (don't forget to substitute valid config_file path there).

With Horovod (highly recommended when using multiple GPUs)::

    mpirun --allow-run-as-root --mca orte_base_help_aggregate 0 -mca btl ^openib -np 4 -H localhost:4 -bind-to none -map-by slot -x LD_LIBRARY_PATH python run.py --config_file=... --mode=train_eval --use_horovod=True --enable_logs

Without Horovod::

    python run.py --config_file=... --mode=train_eval --enable_logs

The description of implemented models is available in the next sections:

Machine translation
-------------------

.. list-table::
   :widths: 1 1 1 1 1
   :header-rows: 1

   * - Config file
     - BLEU
     - Training setup and additional comments
     - Short description of the model
     - Checkpoint
   * - `en-de-nmt-small.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de-nmt-small.py>`_
     - 20.23
     - This model should train on a single GPU such as 1080Ti. It is trained using Adam optimizer.
     - RNN-based. Bi-directional encoder with 2 layers and. GNMT-like decoder with 2 layers and attention. Uses LSTM cells of size 512.
     - `link <https://drive.google.com/file/d/1Ty9hiOQx4V28jJmIbj7FWUyw7LVA39SF/view?usp=sharing>`_
   * - `en-de-gnmt-like-4GPUs.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de-gnmt-like-4GPUs.py>`_
     - 23.89
     - This model was trained on 4 GPUs with Adam optimizer and learning rate decay.
     - RNN-based. This is GNMT-like model which tries to match the one described in https://arxiv.org/abs/1609.08144 as close as possible.
     - `link <https://drive.google.com/file/d/1HVc4S8-wv1-AZK1JeWgn6YNITSFAMes_/view?usp=sharing>`_
   * - `transformer-big.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/transformer-big.py>`_
     - 26.17
     - This model was trained on 4 GPUs with Adam optimizer and learning rate decay.
     - Transformer "big" model. This model does not have any RNN layers
     - `link <https://drive.google.com/file/d/151R6iCCtehRLpnH3nBmhEi_nhNO2mXW8/view?usp=sharing>`_

GNMT model description can be found `here <https://arxiv.org/abs/1609.08144>`_.
Transformer model description can be found `here <https://arxiv.org/abs/1706.03762>`_.
We measure BLEU score on newstest2014.tok.de file using ``multi-bleu.perl`` script from Mosses.

Speech recognition
------------------

Deep Speech 2 based models
~~~~~~~~~~~~~~~~~~~~~~~~~~
Original Deep Speech 2 model description: https://arxiv.org/abs/1512.02595.
The table below contains description and results of
Deep Speech 2 based models available in OpenSeq2Seq.

WER-512 and WER-2048 is word error rate obtained with beam width of 512 and 2048
correspondingly. For beam width of 2048 we also used ``batch_size_per_gpu = 1``
to eliminate the effect of `cudnn padding issue <https://github.com/NVIDIA/OpenSeq2Seq/issues/69>`_.
For more details about model descriptions and training setup,
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
     - 6.12%
     - 5.49%
     - This model was trained for 50 epochs using Adam on the full
       LibriSpeech in a few days using Horovod on four GPUs.
     - This model has 3 convolutional layers and 3 unidirectional
       GRU layers with 1024 units.
     - `link <https://drive.google.com/file/d/1XpnyZzMaO38RE4dSOJZkcaJ3T8B0lxKe/view?usp=sharing>`_
   * - `ds2_small_1gpu.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/ds2_small_1gpu.py>`_
     - 11.77%
     - 9.32%
     - This model was trained for 12 epochs using Adam on a "clean" subset of
       LibriSpeech in less than a day using a single GPU.
     - This model has 2 convolutional layers and 2 bidirectional
       GRU layers with 512 units.
     - `link <https://drive.google.com/file/d/1-OEvxyg7rCogZhejen7pNuKkgvuwCdbk/view?usp=sharing>`_
