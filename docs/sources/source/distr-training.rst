.. _distributed_training:

Multi-GPU and Distributed Training
===================================

.. This section will contain information about how to run different models in
   multi-GPU mode (config change) and in Horovod mode (config change + new command
   to run script). Can also contain some general guidelines for what seems to be
   faster in which cases (e.g. depending on the number of GPUs or input-output
   modality).


OpenSeq2Seq supports two modes for parallel training: `simple multi-tower
approach <https://www.tensorflow.org/programmers_guide/using_gpu#using_multiple_gpus>`_
and `Horovod-based approach <https://github.com/uber/horovod>`_. 


Standard Tensorflow distributed training
------------------------------
For multi-GPU training with native `Distributed Tensorflow approach <https://www.tensorflow.org/deploy/distributed>`_ , 
you  need to set ``use_horovod: False`` and  ``num_gpus=``
in the configuration file. To start training use ``run.py`` script::

    python run.py --config_file=... --mode=train_eval

Horovod
-------
To use Horovod you will need to set ``use_horovod: True`` in the config and `use mpirun <https://github.com/uber/horovod#running-horovod>`_::

    mpiexec -np <num_gpus> python run.py --config_file=... --mode=train_eval --use_horovod=True --enable_logs

You can use Horovod both for multi-GPU and for multi-node training.

.. note::
   ``num_gpus`` parameter will be ignored when ``use_horovod`` is set to True.
   In that case the number of GPUs to use is specified in the command line with
   ``mpirun`` arguments.

.. In general we find it useful to use Horovod mode when ... TODO .




 
