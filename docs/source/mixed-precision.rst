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

How to use mixed precision
--------------------------

...

Implementation details
----------------------

...

Designing new models in MP
--------------------------

...
