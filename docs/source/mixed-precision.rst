Mixed precision training
========================

In this section we will describe

* How to train models in mixed precisionfloat for training on Volta GPUs

* How to port existing models from float32 to mixed precisionfloat
  
* How mixed precision is implemented in our toolkit 

Mixed precision training on Volta GPUs
--------------------------

Half-precision floating numbers (float16) have limited numerical range compared to single-precision float (float32). This constarined numerical range can lead to overflow or underflow during training. We use two techniques to prevent these hazards. First, we  maintain a "master" float32 copy of the weights that accumulates the gradients after each optimizer step.  Second, we use global scaling applied to the loss to prevent the loss of information during backpropagation. We demonstrated that this approach for GNMT(translation) and DeepSpeech2 (speech recognition).  Using this approach, we can reduce the memory consumption of deep learning models by nearly 2x. 

For more details see "Mixed Precision Training" white paper https://arxiv.org/abs/1710.03740 

How to port models from float32 to mixed precision
--------------------------
Enabling mixed precision with existing models in OpenSeq2Seq is easy:
change ``dtype`` parameter of ``model_params`` to "mixed". 
You might need to enable loss scaling: either statically, by setting
``loss_scale`` parameter inside ``model_params`` to the desired number, or
you can use dynamic loss scaling by setting ``automatic_loss_scaling`` parameter
to "Backoff" or "LogMax".

To switch model from float32 to mixed precision, one should add dtype parameter to model description::

  "model_params" = {
    ...
    "dtype": 'mixed',
    ...
  }

One can also set the optional global scale::

  "model_params" = {
    ...
    "dtype": 'mixed',
    "loss_scale": 10.,
    ...
  }

One can also experiment with more fine presion granularity. For example set encoder precison in float16 and decoder in float32::

  "model_params": {
    ...
    "dtype": tf.float16,
    ...
  }
  "decoder_params": {
    ...
    "dtype": tf.float32,  
    ...
  }



Implementation details
----------------------

... This section should also describe loss scaling types? ...

Training in pure float16
----------------------

...

