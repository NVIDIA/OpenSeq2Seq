.. _models_and_recipes:

Models and recipes
==================


.. note::
    Currently OpenSeq2Seq has model implementations for machine translation and
    automatic speech recognition.
    All models work both in float32 and mixed precision.
    We recommend you use :ref:`mixed precision training <mixed_precision>`
    when training on Volta GPUs.


To train models you can use the following commands (don't forget to substitute
valid config_file path there and number of GPUs if using Horovod).

With Horovod (highly recommended when using multiple GPUs)::

    mpiexec --allow-run-as-root -np <num_gpus> python run.py --config_file=... --mode=train_eval --use_horovod=True --enable_logs

Without Horovod::

    python run.py --config_file=... --mode=train_eval --enable_logs

The description of implemented models is available in the next sections:

Machine translation
-------------------

The table below contains description and results of
machine translation models available in OpenSeq2Seq.
Currently, we have GNMT-based model, Transformer-based models and
ConvS2S-based models.

We measure BLEU score on newstest2014.tok.de file using ``multi-bleu.perl`` script from Mosses.
For more details about model descriptions and training setup,
have a look at the `configuration files <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de>`_.


.. list-table::
   :widths: 1 1 1 1 1
   :header-rows: 1

   * - Config file
     - BLEU
     - Training setup and additional comments
     - Short description of the model
     - Checkpoint
   * - `en-de-nmt-small.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/en-de-nmt-small.py>`_
     - 20.23
     - This model should train on a single GPU such as 1080Ti. It is trained using Adam optimizer.
     - RNN-based. Bi-directional encoder with 2 layers and. GNMT-like decoder with 2 layers and attention. Uses LSTM cells of size 512.
     - `link <https://drive.google.com/file/d/1Ty9hiOQx4V28jJmIbj7FWUyw7LVA39SF/view?usp=sharing>`_
   * - `en-de-gnmt-like-4GPUs.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/en-de-gnmt-like-4GPUs.py>`_
     - 23.89
     - This model was trained on 4 GPUs with Adam optimizer and learning rate decay.
     - RNN-based. This is GNMT-like model which tries to match the one described in https://arxiv.org/abs/1609.08144 as close as possible.
     - `link <https://drive.google.com/file/d/1HVc4S8-wv1-AZK1JeWgn6YNITSFAMes_/view?usp=sharing>`_
   * - `transformer-big.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/transformer-big.py>`_
     - 26.17
     - This model was trained on 4 GPUs with Adam optimizer and learning rate decay.
     - Transformer "big" model. This model does not have any RNN layers
     - `link <https://drive.google.com/file/d/151R6iCCtehRLpnH3nBmhEi_nhNO2mXW8/view?usp=sharing>`_
   * - `en-de-convs2s.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2text/en-de/en-de-convs2s.py>`_
     - xx.xx
     - This model was trained on 4 GPUs with Adam optimizer, learning rate decay and warm-up.
     - This is an implementation of the ConvS2S model proposed in https://arxiv.org/abs/1705.03122.
     - Coming soon.

GNMT model description: https://arxiv.org/abs/1609.08144.

Transformer model description: https://arxiv.org/abs/1706.03762.

ConvS2S model description: https://arxiv.org/abs/1705.03122.

Speech recognition
------------------

The table below contains description and results of
speech recognition models available in OpenSeq2Seq.
Currently, we have DeepSpeech2-based models and Wav2Letter-based models.

WER is the word error rate obtained on a dev-clean subset of LibriSpeech using
greedy decoder (``decoder_params/use_language_model = False``).
For the final evaluation we used ``batch_size_per_gpu = 1``
to eliminate the effect of `cudnn padding issue <https://github.com/NVIDIA/OpenSeq2Seq/issues/69>`_.
For more details about model descriptions and training setup,
have a look at the `configuration files <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text>`_.

.. list-table::
   :widths: 1 1 1 1 1
   :header-rows: 1

   * - Config file
     - WER
     - Training setup and additional comments
     - Short description of the model
     - Checkpoint
   * - `ds2_large_8gpus.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/ds2_large_8gpus.py>`_
     - 9.28%
     - This model was trained for 50 epochs using SGD with Momentum and LARC on
       the full LibriSpeech in a few days using Horovod on eight GPUs.
     - This model has 2 convolutional layers and 5 bidirectional
       GRU layers with 800 units.
     - `link <https://drive.google.com/open?id=1EDvL9wMCO2vVE-ynBvpwkFTultbzLNQX>`_
   * - `ds2_medium_4gpus.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/ds2_medium_4gpus.py>`_
     - 22.60%
     - This model was trained for 50 epochs using Adam on the full
       LibriSpeech in a few days using Horovod on four GPUs.
     - This model has 3 convolutional layers and 3 unidirectional
       GRU layers with 1024 units.
     - `link <https://drive.google.com/file/d/1XpnyZzMaO38RE4dSOJZkcaJ3T8B0lxKe/view?usp=sharing>`_
   * - `ds2_small_1gpu.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/ds2_small_1gpu.py>`_
     - 39.08%
     - This model was trained for 12 epochs using Adam on a "clean" subset of
       LibriSpeech in less than a day using a single GPU.
     - This model has 2 convolutional layers and 2 bidirectional
       GRU layers with 512 units.
     - `link <https://drive.google.com/file/d/1-OEvxyg7rCogZhejen7pNuKkgvuwCdbk/view?usp=sharing>`_
   * - `w2l_large_8gpus_mp.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/w2l_large_8gpus_mp.py>`_
     - 7.19%
     - This model was trained for 200 epochs using SGD with Momentum and LARC on
       the full LibriSpeech in three days on eight GPUs using mixed precision.
     - The model has 17 convolutional layers (256--1024 units, 11--29 kernel size).
       We use batch norm between all layers.
     - `link <https://drive.google.com/open?id=140edZXuzehCCaOxgEixJEesvo97EB5i1>`_


Deep Speech 2 model description: https://arxiv.org/abs/1512.02595.

Wav2Letter model description: https://arxiv.org/abs/1609.03193, https://arxiv.org/abs/1712.09444.

Text To Speech
---------------

The table below contains description and results of
text-to-speech models available in OpenSeq2Seq.
Currently, we have a Tacotron2-based model.

.. list-table::
   :widths: 1 1 1 1 1
   :header-rows: 1

   * - Config file
     - Samples
     - Training setup and additional comments
     - Short description of the model
     - Checkpoint
   * - `tacotron_LJ_float.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2speech/tacotron_LJ_float.py>`_
     - Coming soon.
     - Learns magnitude spectrograms. Trained on 1 gpu for 100,000 steps with ADAM.
     - Model tries to match the model description in https://arxiv.org/abs/1712.05884.
       The only difference is that the stop token projection layer is placed after
       the spectrogram projection layer.
     - Coming soon.
   * - `tacotron_LJ_float_8gpu.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2speech/tacotron_LJ_float_8gpu.py>`_
     - Coming soon.
     - Learns magnitude spectrograms. Trained on 8 gpus for 30,000 steps with ADAM and larc.
     - Model tries to match the model description in https://arxiv.org/abs/1712.05884.
       The only difference is that the stop token projection layer is placed after
       the spectrogram projection layer.
     - Coming soon.
   * - `tacotron_LJ_mixed.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2speech/tacotron_LJ_mixed.py>`_
     - Coming soon.
     - Learns magnitude spectrograms. Trained on 1 gpu for 100,000 steps with ADAM and larc.
     - Model tries to match the model description in https://arxiv.org/abs/1712.05884.
       The only difference is that the stop token projection layer is placed after
       the spectrogram projection layer.
     - Coming soon.


Tacotron 2 model description: https://arxiv.org/abs/1712.05884.
