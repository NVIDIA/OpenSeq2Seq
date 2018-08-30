.. _getting-started:

Getting started
===============

OpenSeq2Seq has models for :ref:`machine translation <machine_translation>`, 
:ref:`speech recogtnition <speech_recognition>`, and :ref:`speech synthesis <speech_synthesis>`.
You can find detailed tutorials here:

.. toctree::
   :maxdepth: 2

   getting-started/nmt
   getting-started/asr
   getting-started/tts

All models can be trained in float32 and  :ref:`mixed precision <mixed_precision>` on Volta GPUs.

For multi-GPU and distribuited training we recommended install `Horovod <https://github.com/uber/horovod>`_ . 
When training with Horovod, you should use the following commands (don't forget to substitute 
valid config_file path there and number of GPUs) ::

    mpiexec --allow-run-as-root -np <num_gpus> python run.py --config_file=... --mode=train_eval --use_horovod=True --enable_logs

To train without Horovod::

    python run.py --config_file=... --mode=train_eval --enable_logs
