Text-To-Speech
==================

How to train the model on `LJSpeech <https://keithito.com/LJ-Speech-Dataset/>`_ dataset
----------------------------------------------------------------------------------------

First, you need to download the dataset. The dataset consists of metadata.csv
and a directory of wav files. metadata-csv lists all the wavs filename and their
corresponding transcripts delimited by the '|' character.

In order to train the model, a vocab file must be specified. The vocab file
should contain all the characters present in the dataset plus a special end of
sentence token '~'. The vocab file should have one character per line. An
example vocab file is present inside the openseq2seq/test_utils folder called
"vocab_tts.txt".

Inside the configuration files, be sure to change ``vocab_file``, and
``dataset_location`` to point to the location of the vocab file and the
directory containing the wav files.

The example configuration files assume that the the dataset is split into train,
val, and test sets. you would have to split metadata.csv into three separate
csvs on your own called train.csv, val.csv, and test.csv. You can train the
model via::

    python run.py --config_file=example_configs/text2speech/tacotron_LJ_float.py --mode=train_eval

If you do not want to split the dataset and want to train the model using the
entire dataset, change ``dataset_files`` inside train_params to point to
metadata.csv and run::

    python run.py --config_file=example_configs/text2speech/tacotron_LJ_float.py --mode=train

If you want to run evaluation/inference with the trained model, replace
``--mode=train_eval`` with ``--mode=eval`` or ``--mode=infer``.
For inference you will need to provide additional
``--infer_output_file`` argument. However this argument is ignored to text to
speech. The generated audio filew will be logged to the logdir.
