Mixed precision training
========================

In this section we will describe

* How to train existing models in mixed precision
  (basically, two lines change in the config: dtype and loss scaling).

* How mixed precision is implemented in our toolkit (optimizer and regularizer
  wrappers)

* What are the general steps needed to add mixed precision support to a new
  model. Should also contain example of what we did for our cases: profiling
  GPU/CPU placement, profiling which operations are bottleneck,
  debugging infs/nans, converting some layers to fp32, etc.

Why mixed precision
-------------------

... Should motivate the use of mixed precision: float-16 is great, does not
always work, so using fp-32 weights and loss scaling. Cite the paper. ...

How to use mixed precision
--------------------------

Enabling mixed precision with existing models in OpenSeq2Seq is very easy:
just change ``dtype`` parameter of ``model_params`` dictionary to "mixed". You
might need to additionally enable loss scaling: either statically, by setting
``loss_scale`` parameter inside ``model_params`` to the desired number, or
you can use dynamic loss scaling by setting ``automatic_loss_scaling`` parameter
to "Backoff" or "LogMax".

Implementation details
----------------------

... This section should also describe loss scaling types? ...

Designing new models in MP
--------------------------

...
