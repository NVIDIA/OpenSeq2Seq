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

Layer-wise Adaptive Rate Control (LARC)
---------------------------------------
Disitributed training can be tricky because global batch size increases with the number of nodes.
For large batch training we use Layer-wise Adaptive Rate Control (LARC). The key idea of LARC is to adjust learning rate (LR) for each layer in such way that the magnitude of weight updates would be small compared to weights' norm.  


Neural networks (NN-s) training is based on  Stochastic Gradient Descent (SGD). For example, for the "vanilla" SGD, a mini-batch of *B* samples :math:`x_i` is selected from the training set at each step *t*. Then the stocahtsic gradient :math:`g(t)` of loss function :math:`\nabla L(x_i, w)` wrt weights is computed for a mini-batch: 

.. math::

	g_t = \frac{1}{B} {\sum}_{i=1}^{B} \nabla L(x_i,  w_t)

and then weights *w* are updated based on this stochastic gradient:

.. math::
        
	w_{t+1} = w_t - \lambda * g_t

The standard SGD uses the same LR :math:`\lambda` for all layers. We found that the ratio of the L2-norm of weights and gradients :math:`\frac{| w |}{| g_t |}` varies significantly between weights and biases and between different layers. The ratio is high during the initial phase, and it is rapidly decreasing after few iterations. When :math:`\lambda` is large, the update  :math:`| \lambda * g_t |` can become much larger than  :math:`| w |`, and this can cause divergence. This makes the initial phase of training highly sensitive to the weight initialization and initial LR. 
To stabilize training, we propose to clip the global LR :math:`\gamma` for each layer *k*:

.. math::

    \lambda^k = \min (\gamma, \eta * \frac{| w^k |}{| g^k |} )

where  :math:`\eta < 1` is the LARC "trust" coeffcient. The coeffecient :math:`\eta`  montonically increases with the batch size *B*. 

To use LARC you should add the following lines to model configuration::

  "larc_params": {
    "larc_eta": 0.002,
  }



Notes
~~~~~~~~
The idea of choosing different LR for each layer is known "trick" since 90-s. For example, LeCun etc ["Efficient Backprop" 1998, §4.7] suggested to use larger LR in lower layers than in higher layer, based on the observation that the second derivative of loss function is higher in the upper layers than in small layers. He scaled the global LR for fully-connected layer with *n* of incoming connections by :math:`\frac{1}{\sqrt{n}}`. For convolutional layers, this method would scale global LR for layer *k* with *c* input channels and kernel size :math:`(k \times k)` will be :math:`\lambda_k =  \frac{1}{\sqrt{c}*k}`.



 
