.. _distributed_training:

Distributed training
====================

.. This section will contain information about how to run different models in
   multi-GPU mode (config change) and in Horovod mode (config change + new command
   to run script). Can also contain some general guidelines for what seems to be
   faster in which cases (e.g. depending on the number of GPUs or input-output
   modality).

OpenSeq2Seq supports two modes for distributed training: `simple multi-tower
approach <https://www.tensorflow.org/programmers_guide/using_gpu#using_multiple_gpus>`_
and `Horovod-based approach <https://github.com/uber/horovod>`_. To run the
mutli-GPU training using the first approach, you only need to change
the configuration parameter ``num_gpus``. To use Horovod you will need to set
``use_horovod: True`` in the config and start
``run.py`` script `using mpirun <https://github.com/uber/horovod#running-horovod>`_.
With Horovod you can also enable multi-node execution.

.. note::
   ``num_gpus`` parameter will be ignored when ``use_horovod`` is set to True.
   In that case the number of GPUs to use is specified in the command line with
   ``mpirun`` arguments.

.. In general we find it useful to use Horovod mode when ... TODO .