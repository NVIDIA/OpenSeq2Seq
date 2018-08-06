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
     - 9.28
     - `ds2_large <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/ds2_large_8gpus.py>`_
     - `link <https://drive.google.com/open?id=1EDvL9wMCO2vVE-ynBvpwkFTultbzLNQX>`_

   * - :doc:`Wavel2Letter </speech-recognition/wave2letter>`
     - XXX
     - `w2l_large <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/w2l_large_8gpus.py>`_
     - `link <https://drive.google.com/file/d/151R6iCCtehRLpnH3nBmhEi_nhNO2mXW8/view?usp=sharing>`_


WER is the word error rate obtained on a dev-clean subset of LibriSpeech using
greedy decoder (``decoder_params/use_language_model = False``). 
For the evaluation we used ``batch_size_per_gpu = 1`` 
to eliminate the effect of `cudnn padding issue <https://github.com/NVIDIA/OpenSeq2Seq/issues/69>`_.

For more details about model and training parameters,
have a look at the `configuration files <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   speech-recognition/deepspeech2
   speech-recognition/wave2letter


################
Getting started 
################

Let's build a English speech recogniuiton engine based on DeepSpeech2 model.

********
Get data
******** 

Download and preprocess Librispeech dataset::

./get_librispeech.sh

Download and preprocess OpenSLR mode::



********
Training
********

Next let's train a small DS2 model. 
This model has ...

For this: 

* change ``data_root`` inside ``en-de-nmt-small.py`` to the WMT data location 
* adjust ``num_gpus`` to train on more than one GPU (if available).

Start training::

 python run.py --config_file=example_configs/speech2text/ds2_small_1gpu.py --mode=train_eval

If your GPU does not have enough memory, reduce the ``batch_size_per_gpu``. 
Also, you might want to disable parallel evaluation by using ``--mode=train``. 

*************
Inference
*************

Once training is done (this can take a while on a single GPU), you can run inference::

    python run.py --config_file=example_configs/speech2text/ds2_small_1gpu.py --mode=infer --infer_output_file=ds2_out.txt --num_gpus=1

Also, make sure you use only 1 GPU for inference (``-num_gpus=1``).

*********************************
Computing WER with Language Model
*********************************

Run ```...``` script on:


You should get a WER score around XXXX .





