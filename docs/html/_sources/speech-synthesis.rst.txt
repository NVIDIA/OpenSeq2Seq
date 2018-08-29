.. _speech_synthesis:

Speech Synthesis
================

######
Models
######

Currently we support following models:

.. list-table::
   :widths: 1 2 1 
   :header-rows: 1

   * - Model description
     - Config file
     - Checkpoint
   * - :doc:`Tacotron-2 </speech-synthesis/tacotron-2>`
     - `tacotron_LJ_float.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/18.09/example_configs/text2speech/tacotron_LJ_float.py>`_
     - To be added

The model specification and training parameters can be found in the corresponding config file.

.. toctree::
   :hidden:
   :maxdepth: 1

   speech-synthesis/tacotron-2

################
Getting started 
################

The current tacotron 2 implementation supports the 
`LJSpeech <https://keithito.com/LJ-Speech-Dataset/>`_ dataset. For more details
about the model including hyperparameters and tips, see 
:doc:`Tacotron-2 </speech-synthesis/tacotron-2>`.

********
Get data
********

First, you need to download and extract the dataset into a directory of your
choice. The extracted file should consist of a metadata.csv file and a directory
of wav files. metadata.csv lists all the wav filename and their corresponding
transcripts delimited by the '|' character.


********
Training
********

Next let's train a tacotron 2 model. For this: 

* change ``dataset_location`` under ``data_layer_params`` to point to the
  directory containing the wav files
* change ``dataset_files`` under ``train_params`` to point to metadata.csv

Start training::

    python run.py --config_file=example_configs/text2speech/tacotron_LJ_float.py --mode=train

If your GPU does not have enough memory, reduce the ``batch_size_per_gpu``
parameter.

*************
Inference
*************

Once training is done (this can take a while on a single GPU), you can run
inference. To do some, first create a csv file with lines in the following
format::

    UNUSED | UNUSED | This is an example sentence that I want to generate.

You can put as many lines inside the csv as you want. The model will produce
one audio sample per line and save the audio sample inside your ``log_dir``.
Change ``dataset_files`` under ``infer_params`` to point to your newly created
csv. Lastly, run ::

    python run.py --config_file=example_configs/text2speech/tacotron_LJ_float.py --mode=infer --infer_output_file=unused
