Using existing models
=====================

In this tutorial we will describe everything you can do with OpenSeq2Seq without
writing any new code. We will cover the following topics: how to run one of
the implemented models (for training, evaluation or inference), what parameters
can be specified in the config file/command line and what are the different
kinds of output that OpenSeq2Seq generates for you.

How to run models
-----------------

The main script to run all models is ``run.py``. Since it is a fairly simple
Python script, you can probably understand
how to use it by running ``run.py --help`` which will display all available
command line parameters and their short description. If that does not contain
enough details, continue reading this section. Otherwise, you can safely skip
to the next section, which describes config parameters.

There are 2 main parameters of ``run.py`` that will be
used most often: ``--config_file`` and ``--mode``. The first one is a required
parameter with path to the python configuration file (described in the
:ref:`next section <config-params>`). ``--mode`` parameter can be one of the
"train", "eval", "train\_eval" or "infer". This will do what it says: run
the model in the corresponding mode (with "train\_eval" executing training
with periodic evaluation).
The other parameters of the ``run.py`` script are the following:

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

* ``--debug_port`` --- this enables TensorFlow debugging. To use it first run, e.g.
  ``tensorboard --logdir=. --debugger_port=6067`` and while tensorboard is
  running execute ``run.py`` with ``--debug_port=6067`` attribute.
  After that tensorboard should have debugging tab.

* ``--enable_logs`` --- specifying this parameter will enable additional
  convenient log information to be saved. Namely, the script will save all
  output (both stdout and stderr), exact configuration file, git information
  (git commit hash and git diff) and exact command line parameters used to start
  the script. For all log files it will automatically append current time stamp
  so that subsequent runs do not overwrite any information. One important thing
  to note is that specifying this parameter will force the script to save
  all TensorFlow logs (tensorboard events, checkpoint, etc.) in the ``logs``
  subfolder. Thus, if you want to restore the model that was saved with
  ``enable_logs`` specified you will need to either specify it again or move
  the model checkpoints from the ``logs`` directory into the base ``logdir``
  folder (which is a config parameter).

.. _config-params:

Config parameters
-----------------

The experiment parameters are completely defined in one Python configuration
file. This file must define ``base_params`` dictionary and ``base_model`` class.
``base_model`` should be any class derived from
:class:`Model<models.model.Model>`. Currently it can be
:class:`Speech2Text<models.speech2text.Speech2Text>`,
:class:`Text2Text<models.text2text.Text2Text>` or
:class:`Image2Label<models.image2label.Image2Label>`.
Note that this parameter is not a string, but an actual Python class, so you
will need to add corresponding imports in the configuration file. In addition
to ``base_params`` and ``base_model`` you can define
``train_params``, ``eval_params`` and ``infer_params`` dictionaries that will
overwrite corresponding parts of ``base_params`` when the corresponding mode
is used. For examples of configuration files look in the ``example_configs``
directory. The complete list of all possible configuration parameters is
defined in the documentation in various places. A good place to look first is
the :meth:`Model.__init__()<models.model.Model.__init__>` method
(config parameters section), which defines most of the *first level* parameters:

.. automethod:: models.model.Model.__init__
   :noindex:

Note that some of the parameters are also config dictionaries for corresponding
classes. To see list of their configuration options, you should proceed to the
corresponding class docs. For example, to see all supported data layer parameters,
look into the docs for :class:`data.data_layer.DataLayer`. Sometimes, derived classes
might define their additional parameters, in that case you should be looking
into both, parent class and its child. For example, look into
:class:`models.encoder_decoder.EncoderDecoderModel`, which defines parameters
specific for models that can be expressed as encoder-decoder-loss blocks.
You can also have a look at
:class:`encoders.encoder.Encoder` (which defines some parameters shared across
all encoders) and :class:`encoders.ds2_encoder.DeepSpeech2Encoder` (which
additionally defines a set of DeepSpeech-2 specific parameters).

.. note::
    For convenience all string or numerical config parameters can be overwritten
    by command line arguments. To overwrite parameters of the nested
    dictionaries, separate the dictionary and parameter name with "/".
    For example, try to specify ``--logdir`` argument or
    ``--lr_policy_params/learning_rate`` in your ``run.py`` execution.


What is being logged
--------------------

This section is going to be completed soon.
