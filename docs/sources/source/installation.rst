.. _installation:

Installation instructions
=========================

General installation
--------------------

Clone OpenSeq2Seq  and install Python requirements::

   git clone https://github.com/NVIDIA/OpenSeq2Seq
   cd OpenSeq2Seq
   pip install -r requirements.txt

If you would like to get higher accuracy speech recognition, you have to build TensorFlow from sources as described
in the
:ref:`Installation for speech recognition <installation_speech>`.
Otherwise you can just install TensorFlow using pip::

   pip install tensorflow-gpu


.. _installation_speech:

Installation OpenSeq2Seq for speech recognition
-----------------------------------------------

To obtain the best results for speech recognition it is necessary to
use a CTC beam search decoder with a language model rescoring.
Since TensorFlow does not support it by default, you will need to build TensorFlow
from sources with a custom CTC decoder operation. In order to do that, follow
the steps below. Alternatively, you can disable language model by setting
"use_language_model" parameter of decoder to False, but that will lead to a
worse model accuracy.

How to add CTC decoder with language model to TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install `boost <http://www.boost.org>`_::

    sudo apt-get install libboost-all-dev

2. Build `kenlm <https://github.com/kpu/kenlm>`_ (assuming you are in the
   OpenSeq2Seq folder)::

       ./scripts/install_kenlm.sh

   It will install KenLM in OpenSeq2Seq directory. If you installed KenLM in a different location, 
   you will need to set the corresponding symlink::

        cd OpenSeq2Seq/ctc_decoder_with_lm
        ln -s <kenlm location> kenlm

3. Download and build TensorFlow with custom decoder operation::

        git clone https://github.com/tensorflow/tensorflow
        cd tensorflow
        ./configure
        ln -s <OpenSeq2Seq location>/ctc_decoder_with_lm ./
        bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --copt=-O3  --config=cuda //tensorflow/tools/pip_package:build_pip_package //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //ctc_decoder_with_lm:libctc_decoder_with_kenlm.so
        cp bazel-bin/ctc_decoder_with_lm/*.so ctc_decoder_with_lm/
        bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
        pip install /tmp/tensorflow_pkg/<your tensorflow build>.whl

   Or you can always check the latest TensorFlow
   `installation instructions <https://www.tensorflow.org/install/install_sources>`_,
   except when running bazel build use the following commands instead
   (assuming you are in tensorflow directory)::

        ln -s <OpenSeq2Seq location>/ctc_decoder_with_lm ./
        bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --copt=-O3   --config=cuda //tensorflow/tools/pip_package:build_pip_package //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //ctc_decoder_with_lm:libctc_decoder_with_kenlm.so //ctc_decoder_with_lm:generate_trie
        cp bazel-bin/ctc_decoder_with_lm/*.so ctc_decoder_with_lm/
        cp bazel-bin/ctc_decoder_with_lm/generate_trie ctc_decoder_with_lm/

How to download language model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to achieve the best accuracy, you should download the language
model from `OpenSLR <http://openslr.org/11/>`_ using ``download_lm.sh`` script
(might take some time)::

    ./download_lm.sh

After that you should be able to run toy speech example with enabled CTC beam search decoder::

    python run.py --config_file=example_configs/speech2text/ds2_toy_config.py --mode=train_eval


Horovod installation
--------------------
For multi-GPU and distribuited training we recommended install `Horovod <https://github.com/uber/horovod>`_ .
After TensorFlow and all other requirements are installed,  install mpi:
```pip install mpi4py``` and then follow
`these steps <https://github.com/uber/horovod#install>`_ to install
Horovod.


Running tests
-------------
In order to check that everything is installed correctly it is recommended to
run unittests::

   python -m unittest discover -s open_seq2seq -p '*_test.py'

It might take up to 30 minutes. You should see a lot of output, but no errors
in the end.

Training
--------
To train without Horovod::

    python run.py --config_file=... --mode=train_eval --enable_logs

When training with Horovod, use the following commands (don't forget to substitute
valid config_file path there and number of GPUs) ::

    mpiexec --allow-run-as-root -np <num_gpus> python run.py --config_file=... --mode=train_eval --use_horovod=True --enable_logs


