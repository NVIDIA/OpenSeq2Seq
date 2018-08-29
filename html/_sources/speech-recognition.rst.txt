.. _speech_recognition:

Speech Recognition
==================


######
Models
######

Currently we support following models:

.. list-table::
   :widths: 1 1 2 1
   :header-rows: 1

   * - Model description
     - WER, %
     - Config file
     - Checkpoint

   * - :doc:`DeepSpeech2 </speech-recognition/deepspeech2>`
     - 6.71
     - `ds2_large_mp <https://github.com/NVIDIA/OpenSeq2Seq/blob/18.09/example_configs/speech2text/ds2_large_8gpus_mp.py>`_
     - `link <https://drive.google.com/open?id=1EDvL9wMCO2vVE-ynBvpwkFTultbzLNQX>`_

   * - :doc:`Wavel2Letter+ </speech-recognition/wave2letter>`
     - 6.67
     - `w2l_plus_large_mp <https://github.com/NVIDIA/OpenSeq2Seq/blob/18.09/example_configs/speech2text/w2lplus_large_8gpus_mp.py>`_
     - `link <https://drive.google.com/file/d/10EYe040qVW6cfygSZz6HwGQDylahQNSa/view?usp=sharing>`_


WER is the word error rate obtained on a dev-clean subset of LibriSpeech using
greedy decoder (``decoder_params/use_language_model = False``).
For the evaluation we used ``batch_size_per_gpu = 1``
to eliminate the effect of `cudnn padding issue <https://github.com/NVIDIA/OpenSeq2Seq/issues/69>`_.

For more details about model and training parameters,
have a look at the `configuration files <https://github.com/NVIDIA/OpenSeq2Seq/blob/18.09/example_configs/speech2text>`_ and specific model's documentation.

.. toctree::
   :hidden:
   :maxdepth: 1

   speech-recognition/deepspeech2
   speech-recognition/wave2letter


################
Getting started
################

You can start with :doc:`these instructions </speech-recognition/get_started_toy_model>`
to play with a very small model on a toy dataset.

Now let's consider a relatively lightweight version of DeepSpeech2 based model for
English speech recognition on LibriSpeech dataset.

********
Get data
********

Download and preprocess Librispeech dataset::

 python scripts/import_librivox.py

Download and preprocess OpenSLR language model::

 scripts/download_lm.sh


********
Training
********

Let's train a small DS2 model.

This model can be trained on 12 GB GPU within a day.

Start training::

 python run.py --config_file=example_configs/speech2text/ds2_small_1gpu.py --mode=train_eval

If your GPU does not have enough memory, reduce the ``batch_size_per_gpu``.
Also, you might want to disable evaluation during training by using ``--mode=train``.


**********
Evaluation
**********

In order to get greedy Word Error Rate (WER) metric on validation dataset, please run the following command::

 python run.py --config_file=example_configs/speech2text/ds2_small_1gpu.py --mode=eval


If you would like to use beam search decoder with language model re-scoring, please use parameter ``decoder_params/use_language_model=True``::

 python run.py --config_file=example_configs/speech2text/ds2_small_1gpu.py --mode=eval --decoder_params/use_language_model=True


*************
Inference
*************

Once training is done (this can take a while on a single GPU), you can run inference::

 python run.py --config_file=example_configs/speech2text/ds2_small_1gpu.py --mode=infer --infer_output_file=ds2_out.txt

