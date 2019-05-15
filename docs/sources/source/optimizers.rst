.. _optimizers:

Optimizers
===================================

.. This section contain information about LARC and NovoGrad.


OpenSeq2Seq supports two new optimizers: LARC and NovoGrad. 


Layer-wise Adaptive Rate Control (LARC)
---------------------------------------
The key idea of LARC is to adjust learning rate (LR) for each layer in such way that the magnitude of weight updates would be small compared to weights' norm.  


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


NovoGrad
--------
NovoGrad is a first-order SGD-based algorithm, which computes second moments per layer instead of per weight as in Adam. Compared to Adam, NovoGrad takes less   memory, and we find it to be more numerically stable.

NovoGrad computes the stochastic gradient :math:`g_t` at each step *t*. Then the second-order moment :math:`v^l_t` is computed for each layer *l*, similar to ND-Adam (Zhang 2017):

.. math::

    v^l_t = \beta_2 \cdot v^l_{t-1} + (1-\beta_2) \cdot ||g^l_t||^2

The moment :math:`v^l_t` is used to re-scale gradients :math:`g^l_t` before calculating the first-order moment :math:`m^l_t`:

.. math::

    m^l_t = \beta_1 \cdot m^l_{t-1} +  \frac{g^l_t}{\sqrt{v^l_t} +\epsilon}


where  :math:`\lambda_t` is the current global learning rate. If L2-regularization is used, a weight decay term :math:`d \cdot w^l_{t-1}` is added to the re-scaled gradient (as in AdamW, Loshchilov 2017): 

.. math::

    m^l_t = \beta_1 \cdot m^l_{t-1} +  (\frac{g^l_t}{\sqrt{v^l_t} + \epsilon} + d \cdot w^l_{t-1})

Finally, new weights are computed using 

.. math::
    w_t = w_{t-1} - \alpha_t \cdot m_t 


To use Novograd you should tun off the standard regularization and add the following lines to model configuration::

    "optimizer": NovoGrad,
    "optimizer_params": {
        "beta1": 0.95,
        "beta2": 0.98,
        "epsilon": 1e-08,
        "weight_decay": 0.001,
    },



References
~~~~~~~~~~
1. Zhang Z.,  Ma L., Li Z., and  Wu C.,  Normalized direction-preserving Adam. arXiv e-prints, arXiv:1709.0454, 2018
2. Loshchilov I. and Hutter F., Fixing weight decay regularization in Adam.   arXiv e-prints, arXiv:1711.0510, 2017

 
