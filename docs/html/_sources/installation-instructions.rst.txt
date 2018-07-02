.. _installation-instructions:

Installation instructions
=========================

General installation
--------------------

The simplest way to obtain complete and working version of OpenSeq2Seq is by
using our docker container: TODO: add link.

If you are using docker, you can skip the rest of this page. Otherwise, follow
the instructions below to have everything set up::

   git clone https://github.com/NVIDIA/OpenSeq2Seq
   cd OpenSeq2Seq
   pip install -r requirements.txt

Now, if you are not going to use speech-to-text models, you can just install
TensorFlow using pip::

   pip install tensorflow-gpu

and skip the next section that describes speech-specific installation process.
If you need speech models, then you have to build TensorFlow from sources as described 
in the next section.

Running tests
-------------
In order to check that everything is installed correctly it is recommended to
run unittests::

   python -m unittest discover -s open_seq2seq -p '*_test.py'

It might take up to 30 minutes. You should see a lot of output, but no errors
in the end.

.. _installation_speech:

OpenSeq2Seq for speech
----------------------

To obtain the best results for automatic speech recognition it is necessary to
use CTC decoder with language model. Since TensorFlow does not support it by
default, you will need to build TensorFlow from sources with
custom CTC decoder with language model operation. In order to do that, follow
the steps below. Alternatively, you can disable language model by setting
"use_language_model" parameter of decoder to False, but that will lead to a much
worse model accuracy.

How to add CTC decoder with language model to TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install `boost <http://www.boost.org>`_::

    sudo apt-get install libboost-all-dev

2. Build `kenlm <https://github.com/kpu/kenlm>`_ (assuming you are in the
   OpenSeq2Seq folder)::

       cd ctc_decoder_with_lm
       git clone https://github.com/kpu/kenlm
       cd kenlm
       mkdir -p build
       cd build
       cmake ..
       make -j 

   If you prefer to build kenlm in different location, you will need to set
   the corresponding symlink::

        cd OpenSeq2Seq/ctc_decoder_with_lm
        ln -s <kenlm location> kenlm

3. Download and build TensorFlow with custom decoder operation:

   On Ubuntu 16.04 the following sequence of commands should work.
   At the location that you want to install TensorFlow to, run::

        git clone https://github.com/tensorflow/tensorflow
        cd tensorflow
        ./configure
        ln -s <OpenSeq2Seq location>/ctc_decoder_with_lm ./
        bazel build -c opt --copt=-O3 --config=cuda //tensorflow/tools/pip_package:build_pip_package //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //ctc_decoder_with_lm:libctc_decoder_with_kenlm.so //ctc_decoder_with_lm:generate_trie
        cp bazel-bin/ctc_decoder_with_lm/*.so ctc_decoder_with_lm/
        cp bazel-bin/ctc_decoder_with_lm/generate_trie ctc_decoder_with_lm/
        bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
        pip install /tmp/tensorflow_pkg/<your tensorflow build>.whl

   If you are not on Ubuntu 16.04 or if something does not work, try to follow
   the usual TensorFlow
   `installation instructions <https://www.tensorflow.org/install/install_sources>`_,
   except when running bazel build use the following commands instead
   (assuming you are in tensorflow directory)::

        ln -s <OpenSeq2Seq location>/ctc_decoder_with_lm ./
        bazel build -c opt --copt=-O3  --config=cuda //tensorflow/tools/pip_package:build_pip_package //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //ctc_decoder_with_lm:libctc_decoder_with_kenlm.so //ctc_decoder_with_lm:generate_trie
        cp bazel-bin/ctc_decoder_with_lm/*.so ctc_decoder_with_lm/
        cp bazel-bin/ctc_decoder_with_lm/generate_trie ctc_decoder_with_lm/

How to download language model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to achieve the best accuracy, you should download the big language
model from `Mozilla's DeepSpeech repository <https://github.com/mozilla/DeepSpeech/tree/master/data/lm>`_.
This can be done by running ``download_lm.sh`` script
(might take some time, since the size of the model is around 1.5Gb)::

    ./download_lm.sh

If the script fails with 503 Error, try running at again or download "lm.binary"
and "trie" files manually from Mozilla's GitHub and put them in language_model
folder.

After that you should be able to run toy speech example with no errors::

    python run.py --config_file=example_configs/speech2text/ds2_toy_data_config.py --mode=train_eval

Horovod installation
--------------------

After TensorFlow and all other requirements are installed, you can also follow
`these steps <https://github.com/uber/horovod#install>`_ to enable
Horovod-based distributed training. You also need to install:
```pip install mpi4py```

