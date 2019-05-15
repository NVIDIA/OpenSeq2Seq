.. _image_classification:

Image Classification
====================

######
Model
######

Our ResNet-50 v2 model is a mixed precison replica of `TensorFlow ResNet-50 <https://github.com/tensorflow/models/tree/master/official/resnet>`_ , which corresponds to the model defined in the paper `Identity Mappings in Deep Residual Networks <https://arxiv.org/abs/1603.05027>`_ by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

This model was trained with different optimizers to state-of-the art accuracy for ResNet-50 model. 
Our best model reached top-1=77.63%, top-5=93.73 accuracy for Imagenet classification task. 


################
Get data
################

You will need to download the ImageNet dataset and convert it to TFRecord format as described in
`TensorFlow ResNet <https://github.com/tensorflow/models/tree/master/official/resnet`_


################
Training
################

Let's train a model using SGD with momentum. To train model with 1 GPU with float precision::

 python run.py --config_file=example_configs/image2label/resnet-50-v2.py --mode=train_eval

If your GPU does not have enough memory, you can reduce the ``batch_size_per_gpu``::
 
 python run.py --config_file=example_configs/image2label/resnet-50-v2.py --mode=train_eval --batch_size_per_gpu=32

******************
Multi-GPU training
******************

If you have 2 GPUs, then you can use "native" Tensorflow multi-GPU training by setting ``num_gpus``::

 python run.py --config_file=example_configs/image2label/resnet-50-v2.py --mode=train_eval --use_horovod=False --num_gpus=2

or you can use Horovod (``-np`` flag defines number of GPUs)::

 mpirun --allow-run-as-root --mca orte_base_help_aggregate 0 -mca btl ^openib -np 2 -H localhost:8 -bind-to none --map-by slot -x LD_LIBRARY_PATH python run.py --config_file=example_configs/image2label/resnet-50-v2.py --mode=train_eval --use_horovod=True

*****************************
Training with Mixed Precicion
*****************************
If you have Volta or Turing GPU which supports float16, you can speed-up training by using mixed precision::

 python run.py --config_file=example_configs/image2label/resnet-50-v2-mp.py --mode=train_eval --use_horovod=False --num_gpus=2


############
Checkpoints
############

We have trained ResNet-50 with 3 optimizers:

 * `SGD with momentum <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/image2label/resnet-50-v2-mp.py>`_ 

 * `AdamW <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/image2label/resnet-50v2-adamw.py>`_ 

 * `NovoGrad <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/image2label/resnet-50v2-nvgrad.py>`_ 

.. list-table::
   :widths: 2 1 1 1 2 1 1
   :header-rows: 1

   * - Optimizer
     - Training epochs
     - top-1, %
     - top-5, %
     - Config file
     - Checkpoint
     - log

   * - SGD with momentum
     - 100
     - 76.38
     - 93.08
     - `sgd_100 <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/image2label/resnet-50-v2-mp.py>`_
     - checkpoint 
     - log

   * - AdamW
     - 100
     - 76.36 
     - 93.01
     - `adamw_100 <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/image2label/resnet-50v2-adamw.py>`_ 
     - checkpoint 
     - log

   * - :doc:`NovoGrad </optimizers>`
     - 100
     - 77.00
     - 93.37 
     - `nvgrad_100 <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/image2label/resnet-50v2-nvgrad.py>`_ 
     - checkpoint 
     - log

   * - :doc:`NovoGrad </optimizers>`
     - 300
     - 77.63
     - 93.73 
     - `nvgrad_300 <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/image2label/resnet-50v2-nvgrad.py>`_ 
     - checkpoint
     - log

Detailed training parameters are in corresponding configuration file.






