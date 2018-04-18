Using existing models
=====================

In this tutorial we will describe everything you can do with OpenSeq2Seq without
writing any new code. We will cover the following topics: how to run one of
the implemented models (for training, evaluation or inference), what parameters
can be specified in the config file/command line and what are the different
kinds of output that OpenSeq2Seq generates for you.

How to run models
-----------------

There are two scripts that can be used to run the models: ``run.py`` and
``start_experiment.sh``. The latter one is just a convenient bash
wrapper around ``run.py`` that adds additional functionality, so we will
describe it in the end. Since ``run.py`` is a fairly simple Python script,
you can probably understand
how to use it by running ``run.py --help`` which will display all available
command line parameters and their short description. If that does not contain
enough details, continue reading this section. Otherwise, just skip to the
description of ``start_experiment.sh`` script (last paragraph of this section).

There are 2 main parameters of ``run.py`` that will be
used most often: ``--config_file`` and ``--mode``. The first one is a required
parameter with path to the python configuration file (described in the :ref:`next
section <config-params>`). ``--mode`` parameter can be one of the "train",
"eval", "train\_eval" or "infer". This will do what it says: run the model in
the corresponding mode (with "train\_eval" executing training with periodic
evaluation). The other parameters of the ``run.py`` script are the following:

* ``--continue_learning`` --- specify this when you want to continue learning
  from existing checkpoint. This parameter is only checked when ``--mode`` is
  "train" or "train\_eval".

* ``--infer_output_file`` --- this specifies the path to output of the inference.
  This parameter is only checked when ``--mode`` is "infer".

* ``--no_dir_check`` --- this parameter disables log directory checking.
  By default, ``run.py`` will be checking that the log
  directory (specified in the python config) is valid. Specifically, it will
  check that it exists when ``--mode`` equals "eval" or "infer"
  (or when ``--continue_learning`` is specified for training). If training is
  performed, but ``--continue_learning`` is not specified, the script will check
  that log directory is empty or does not exist, otherwise finishing with
  exception. Finally, whenever necessary it will check that the log directory
  contains a valid TensorFlow checkpoint of the saved model.

* ``--benchmark`` --- specifying this parameter will automatically prepare config
  for time benchmarking: disable all logging and evaluation. This parameter is
  only useful for training benchmarking, since in other cases no config
  preparation is needed. Moreover, specifying it will force the model to run
  in the "train" mode.

* ``--bench_steps`` --- number of steps to run the model for benchmarking. For
  now this can only be used in conjunction with ``--benchmark`` parameter and
  thus only works in the training benchmarking.

* ``--bench_start`` --- first step to start counting time for benchmarking. This
  parameter works in all modes whether or not ``--benchmark`` parameter was
  specified.

In order to make it more convenient to run multiple experiments we provide
``start_experiment.sh`` script that is a wrapper around ``run.py`` script which
does the following things. First, it will make sure that the complete output of
``run.py`` is saved inside the log directory (in the "output\_<time stamp>.log
file, where <time stamp> is a string with current date and time to make sure
that everything is saved if you run this script multiple times).
Second, it will log the current git commit and git diff in the
"gitinfo\_<time stamp>.log" file to make it possible to completely reproduce the
experiment. Finally, it will save the current experiment configuration in the
"config\_<time stamp>.py" file. To run ``start_experiment.sh`` you will need to
define the following environment variables: ``LOGDIR`` (path to the desired log
directory), ``CONFIG_FILE`` (path to the Python configuration file), ``MODE``
(mode to execute ``run.py`` in) and ``CONTINUE_LEARNING`` (whether to specify
``--continue_learning`` flag for ``run.py``, could be 1 or 0). For example to
train DeepSpeech2-like model on the toy speech data you can run::

   LOGDIR=experiments/librispeech CONFIG_FILE=example_configs/speech2text/ds2_toy_data_config.py MODE=train_eval CONTINUE_LEARNING=0 ./start_experiment.sh

.. _config-params:
Config parameters
-----------------

The experiment parameters are completely defined in one Python configuration
file. This file must define ``base_params`` dictionary and can define additional
``train_params``, ``eval_params`` and ``infer_params`` dictionaries that will
overwrite corresponding parts of ``base_params`` when the corresponding mode
is used. Here is an example of configuration file for the speech-to-text model:

.. literalinclude:: ../../../example_configs/speech2text/ds2_librispeech_adam_config.py
   :linenos:

That's a big file with a lot of parameters, but you will rarely need to write it
yourself from scratch. Most of the time you can just copy one of the example
configs and make a few lines modification to customize it for your specific
problems. So let's walk-through this file to make sure you understand all the
possible configuration parameters.

Since the configuration file is just a regular Python file, it starts with a
series of imports (lines 1--7). Then the main configuration dictionary
``base_params`` is defined which has several groups of parameters. The first
group (lines 11--23) is the general experiment configuration parameters, such as
random seed, number of GPUs to use, batch size per GPU, etc. Most of these
parameters are self-explanatory, ...

Text-to-text specifics
~~~~~~~~~~~~~~~~~~~~~~~~

...

Speech-to-text specifics
~~~~~~~~~~~~~~~~~~~~~~~~

...

What is being logged
--------------------

...
