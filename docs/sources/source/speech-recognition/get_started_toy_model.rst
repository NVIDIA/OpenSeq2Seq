Speech Recognition
==================

First, make sure you followed the :ref:`Speech installation instructions <installation_speech>` .

After that you should be able to run toy speech example with no errors::

    python run.py --config_file=example_configs/speech2text/ds2_toy_data_config.py --mode=train_eval


How to train the model on `LibriSpeech <http://www.openslr.org/12>`_ dataset
----------------------------------------------------------------------------

First, you need to download and preprocess the LibriSpeech dataset.
Assuming you are in the base folder, run::

    sudo apt-get -y install sox libsox-dev
    mkdir -p data
    python import_librivox.py data/librispeech

Note, that this will take a lot of time, since
it needs to download, extract and convert around 55GB of audio files. The final
dataset size will be around 224GB (including archives and original compressed audio files, feel free to delete them to get 106GB).

Now, everything should be setup to train the model::

    python run.py --config_file=example_configs/speech2text/ds2_librispeech_larc_config.py --mode=train_eval

If you want to run evaluation/inference with the trained model, replace
``--mode=train_eval`` with ``--mode=eval`` or ``--mode=infer``.
For inference you will need to provide additional
``--infer_output_file=<output file>`` argument.

How to build your own language model
------------------------------------

Language models usually help speech2text decoder to correct misspellings in recognized utterances.
:class:`FullyConnectedCTCDecoder <open_seq2seq.decoders.fc_decoder.FullyConnectedCTCDecoder>` uses N-gram `KenLM <https://github.com/kpu/kenlm>`_ based models.
In order to build a language model, please use ``build_lm.py`` script.
For example, run the following commands for LibriSpeech::

    export LS_DIR=/data/speech/LibriSpeech/
    python build_lm.py --n 5 $LS_DIR/librivox-train-clean-100.csv $LS_DIR/librivox-train-clean-360.csv librivox-train-other-500.csv

You will get as a result two files: the binary language model ``librivox-train-clean-100.binary`` and its trie ``librivox-train-clean-100.trie``.


