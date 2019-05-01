.. _speech_recognition:

Speech Recognition
==================

######
Models
######

Currently we support following models:

.. list-table::
   :widths: 2 1 2 1
   :header-rows: 1

   * - Model description
     - Greedy WER, %
     - Config file
     - Checkpoint

   * - :doc:`DeepSpeech2 </speech-recognition/deepspeech2>`
     - 6.71
     - `ds2_large_mp <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/ds2_large_8gpus_mp.py>`_
     - `link <https://drive.google.com/open?id=1EDvL9wMCO2vVE-ynBvpwkFTultbzLNQX>`_

   * - :doc:`Wave2Letter+ </speech-recognition/wave2letter>`
     - 6.67
     - `w2l_plus_large_mp <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/w2lplus_large_8gpus_mp.py>`_
     - `link <https://drive.google.com/file/d/10EYe040qVW6cfygSZz6HwGQDylahQNSa/view?usp=sharing>`_

   * - :doc:`Jasper DR 10x5 </speech-recognition/jasper>`
     - 3.64
     - `jasper10x5_LibriSpeech_nvgrad <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/jasper10x5_LibriSpeech_nvgrad.py>`_
     - `link <https://drive.google.com/a/nvidia.com/file/d/1gzGT8HoVNKY1i5HNQTKaSoCu7JHV4siR/view?usp=sharing>`_


WER is the word error rate obtained on a dev-clean subset of LibriSpeech using
greedy decoder (``decoder_params/use_language_model = False``).
For the evaluation we used ``batch_size_per_gpu = 1``
to eliminate the effect of `cuDNN padding issue <https://github.com/NVIDIA/OpenSeq2Seq/issues/69>`_.

For more details about model and training parameters,
have a look at the `configuration files <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text>`_ and specific model's documentation.

.. toctree::
   :hidden:
   :maxdepth: 1

   speech-recognition/deepspeech2
   speech-recognition/wave2letter
   speech-recognition/jasper


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

Download and preprocess LibriSpeech dataset::

 python scripts/import_librivox.py data/librispeech

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

If you would like to use beam search decoder with language model re-scoring, please see `link <https://github.com/NVIDIA/OpenSeq2Seq/tree/master/external_lm_rescore>`_


*************
Inference
*************

Once training is done (this can take a while on a single GPU), you can run inference::

 python run.py --config_file=example_configs/speech2text/ds2_small_1gpu.py --mode=infer --infer_output_file=ds2_out.txt



******************
Multi-GPU training
******************

To train on <N> GPUs without Horovod::

    python run.py --config_file=... --mode=train_eval --use_horovod=False --num_gpus=<N>

To train with Horovod on <N> GPUs, use the following command::

    mpiexec --allow-run-as-root -np <N> python run.py --config_file=... --mode=train_eval --use_horovod=True

##############
Synthetic data
##############

Speech recognition models can be optionally trained using synthetic data.
The creation of the synthetic data and training process is described :ref:`here <synthetic_data>`.


.. toctree::
   :hidden:
   :maxdepth: 1

   speech-recognition/synthetic_dataset


##############
Tools
##############
Word alignments : :doc:`Align words </speech-recognition/speech-to-text-align>`
