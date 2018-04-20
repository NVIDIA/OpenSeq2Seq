# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import abc
import six
import tensorflow as tf
import copy

from open_seq2seq.utils.utils import deco_print
from open_seq2seq.optimizers import optimize_loss, get_regularization_loss
from open_seq2seq.utils.utils import check_params


@six.add_metaclass(abc.ABCMeta)
class Model:
  """Abstract class that any model should inherit from.
  It automatically enables multi-GPU (or Horovod) computation,
  has mixed precision support, logs training summaries, etc.
  """
  @staticmethod
  def get_required_params():
    """Static method with description of required parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **have to** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return {
      'learning_rate': float,
      'optimizer': None,  # could be class or string
      'logdir': str,
      'batch_size_per_gpu': int,
    }

  @staticmethod
  def get_optional_params():
    """Static method with description of optional parameters.

      Returns:
        dict:
            Dictionary containing all the parameters that **can** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
    return {
      'optimizer_params': dict,
      'initializer': None,  # any valid TensorFlow initializer
      'initializer_params': dict,
      'regularizer': None,  # any valid TensorFlow regularizer
      'regularizer_params': dict,
      'dtype': [tf.float16, tf.float32, 'mixed'],
      'lr_policy': None,  # any valid learning rate policy function
      'lr_policy_params': dict,
      'max_grad_norm': float,
      'larc_nu': float,
      'larc_mode': ['scale', 'clip'],
      'loss_scale': float,
      'automatic_loss_scaling': [None, 'Backoff', 'LogMax'],
      'summaries': list,
      'max_steps': int,
      'num_epochs': int,
    }

  def __init__(self,
               params,
               data_layer,
               global_step=None,
               force_var_reuse=False,
               mode="train",
               gpu_ids=None,
               hvd=None):
    """Model constructor. The TensorFlow graph is created here.

    Args:
      params (dict): parameters describing the model.
          All supported parameters are listed in :meth:`get_required_params`,
          :meth:`get_optional_params` functions.
      data_layer (DataLayer): The :class:`DataLayer` instance to take data from.
      global_step (optional): TensorFlow global step or None
      force_var_reuse (bool, optional): if true, all variables will be re-used.
          Useful for creating evaluation model alongside the training model or
          for multi-GPU training.
      mode (string, optional): "train", "eval" or "infer".
          If mode is "train" all parts of the graph will be built
          (model, loss, optimizer).
          If mode is "eval", only model and loss will be built.
          If mode is "infer", only model will be built.
      gpu_ids (list, optional): a list of gpu ids to run the model on.
          For distributed training using Horovod this parameter is ignored.

    Config parameters:

    * **learning_rate** (float) --- initial learning rate for training.
    * **optimizer** (string or TensorFlow optimizer class) --- optimizer to
      use for training. Could be either "Adam", "Adagrad", "Ftrl", "Momentum",
      "RMSProp", "SGD" or any valid TensorFlow optimizer class.
    * **optimizer_params** (dict) --- dictionary that will be passed to
      optimizer ``__init__`` method.
    * **initializer** --- any valid TensorFlow initializer.
    * **initializer_params** (dict) --- dictionary that will be passed to
      initializer ``__init__`` method.
    * **regularizer** --- and valid TensorFlow regularizer.
    * **regularizer_params** (dict) --- dictionary that will be passed to
      regularizer ``__init__`` method.
    * **dtype** --- model dtype. Could be either ``tf.float16``, ``tf.float32``
      or "mixed". For details see
      :ref:`mixed precision training <mixed_precision>` section in docs.
    * **lr_policy** --- any valid learning rate policy function. For examples,
      see :any:`optimizers.lr_policies` module.
    * **lr_policy_params** (dict) --- dictionary containing lr_policy
      parameters.
    * **max_grad_norm** (float) --- maximum value of gradient norm. Clipping
      will be performed if some gradients exceed this value (this is checked
      for each variable independently).
    * **larc_mode** --- specify this to use LARC or LARS optimization
      algorithms. Could be either "scale" (LARS) or "clip" (LARC).
      You also need to specify ``larc_nu`` to enable LARC or LARS. Note that
      it works in addition to any other optimization algorithm since we treat
      it as adaptive gradient clipping and learning rate adjustment.
    * **larc_nu** (float) --- LARC or LARS scaling parameter.
    * **loss_scale** (float) --- static loss scale to use. For details see
      :ref:`mixed precision training <mixed_precision>` section in docs.
    * **autimatic_loss_scaling** --- automatic loss scaling mode. Could be
      either None, "Backoff" or "Logmax". For details see
      :ref:`mixed precision training <mixed_precision>` section in docs.
    * **summaries** (list) --- which summaries to log. Could contain
      "learning_rate", "gradients", "gradient_norm", "global_gradient_norm",
      "variables", "variable_norm".
    """
    check_params(params, self.get_required_params(), self.get_optional_params())

    self._on_horovod = hvd is not None
    self._params = copy.deepcopy(params)
    self._dtype = self._params.get("dtype", tf.float32)

    if self._on_horovod:
        self._num_gpus = 1
    else:
      self._num_gpus = len(gpu_ids)

    self._outputs = [None] * self.num_gpus
    self._data_layer = data_layer
    input_tensors = data_layer.get_input_tensors()

    self._mode = mode
    if self._mode not in ["train", "infer", "eval"]:
      raise ValueError("Unknown mode")

    if global_step is not None:
      self.global_step = global_step
    else:
      self.global_step = tf.train.get_or_create_global_step()

    if 'initializer' not in self.params:
      initializer = None
    else:
      init_dict = self.params.get('initializer_params', {})
      initializer = self.params['initializer'](**init_dict)

    if not self._on_horovod:  # not using Horovod
      # below we follow data parallelism for multi-GPU training
      # actual per GPU data feeds
      losses = []
      for gpu_ind, gpu_id in enumerate(gpu_ids):
        with tf.device("/gpu:{}".format(gpu_id)), tf.variable_scope(
          name_or_scope=tf.get_variable_scope(),
          # re-using variables across GPUs.
          reuse=force_var_reuse or (gpu_ind > 0),
          initializer=initializer,
          dtype=self.get_tf_dtype(),
        ):
          deco_print("Building graph on GPU:{}".format(gpu_id))

          loss, self._outputs[gpu_ind] = self._build_forward_pass_graph(
            [input_tensor[gpu_ind] for input_tensor in input_tensors],
            gpu_id=gpu_ind,
          )
          if self._mode == "train" or self._mode == "eval":
            losses.append(loss)
      # end of for gpu_ind loop
      if self._mode == "train" or self._mode == "eval":
        self.loss = tf.reduce_mean(losses)
    else:  # is using Horovod
      # gpu_id should always be zero, since Horovod takes care of isolating
      # different processes to 1 GPU only
      with tf.device("/gpu:0"), tf.variable_scope(
          name_or_scope=tf.get_variable_scope(),
          reuse=force_var_reuse,
          initializer=initializer,
          dtype=self.get_tf_dtype(),
      ):
        deco_print("Building graph in Horovod rank: {}".format(hvd.rank()))
        loss, self._outputs[0] = self._build_forward_pass_graph(input_tensors,
                                                                gpu_id=0)
        if self._mode == "train" or self._mode == "eval":
          self.loss = loss

    if self._mode == "train":
      if "max_steps" in self.params:
        self._last_step = self.params["max_steps"]
        self._step_size = None
      else:
        ds_sample_size = self.data_layer.get_size_in_samples()
        total_bs = self.data_layer.params['batch_size']
        if hvd:
          total_bs *= hvd.size()
        # doing a few less steps if data size is not divisible by the batch size
        self._step_size = ds_sample_size // total_bs
        self._last_step = self.params['num_epochs'] * self._step_size

      if 'lr_policy' not in self.params:
        lr_policy = None
      else:
        lr_params = self.params.get('lr_policy_params', {})
        # adding default decay_steps = max_steps if lr_policy supports it and
        # different value is not provided
        if 'decay_steps' in self.params['lr_policy'].__code__.co_varnames and \
           'decay_steps' not in lr_params:
          lr_params['decay_steps'] = self._last_step
        lr_policy = lambda lr, gs: self.params['lr_policy'](lr, gs, **lr_params)

      self.train_op = optimize_loss(
        loss=self.loss + get_regularization_loss(),
        dtype=self.params['dtype'],
        global_step=self.global_step,
        learning_rate=self.params['learning_rate'],
        optimizer=self.params['optimizer'],
        optimizer_params=self.params.get('optimizer_params', {}),
        gradient_noise_scale=None,
        gradient_multipliers=None,
        clip_gradients=self.params.get('max_grad_norm', None),
        learning_rate_decay_fn=lr_policy,
        update_ops=None,
        variables=None,
        name="Loss_Optimization",
        summaries=self.params.get('summaries', None),
        colocate_gradients_with_ops=True,
        increment_global_step=True,
        LARC_nu=self.params.get('larc_nu', None),
        LARC_mode=self.params.get('larc_mode', 'clip'),
        loss_scale=self.params.get('loss_scale', 1.0),
        automatic_loss_scaling=self.params.get('automatic_loss_scaling', None),
        on_horovod=self._on_horovod
      )
      tf.summary.scalar(name="train_loss", tensor=self.loss)

      if not self._on_horovod or hvd.rank() == 0:
        deco_print("Trainable variables:")
        total_params = 0
        unknown_shape = False
        for var in tf.trainable_variables():
          var_params = 1
          deco_print('{}'.format(var.name), offset=2)
          deco_print('shape: {}, {}'.format(var.get_shape(), var.dtype),
                     offset=4)
          if var.get_shape():
            for dim in var.get_shape():
              var_params *= dim.value
            total_params += var_params
          else:
            unknown_shape = True
        if unknown_shape:
          deco_print("Encountered unknown variable shape, can't compute total "
                     "number of parameters.")
        else:
          deco_print('Total trainable parameters: {}'.format(total_params))

  @abc.abstractmethod
  def _build_forward_pass_graph(self, input_tensors, gpu_id=0):
    """Abstract method. Should create the graph of the forward pass of the model.

    Args:
      input_tensors (list): list of all input tensors
          required to build the model.
      gpu_id (int, optional): id of the GPU where the current copy of the model
          is constructed. For Horovod this is always zero.

    Returns:
      tuple: tuple containing loss tensor and samples tensor.

      Loss tensor will be automatically provided to the optimizer and
      corresponding :attr:`train_op` will be created.

      Samples tensors are stored in the :attr:`_outputs` attribute and can be
      accessed by calling :meth:`get_output_tensors` function. For example,
      this happens inside :class:`utils.hooks.RunEvaluationHook`
      to fetch output values for evaluation.

      Both loss and samples can be None when corresponding part of the graph
      is not built.
    """
    pass

  def maybe_print_logs(self, input_values, output_values):
    """This function can be used to print logs that help to visualize training.
    For example, you can print sample input sequences and their corresponding
    predictions. This function will be called every ``print_samples_steps``
    (config parameter) iterations and input/output values will be populated
    automatically by calling ``sess.run`` on corresponding tensors. Note that
    this function is not abstract and does not have to be implemented in
    derived classes. But if additional printing functionality is required,
    overwriting this function can be a useful way to add it.

    Args:
      input_values: evaluation of :meth:`self.data_layer.get_input_tensors()
                                  <data.data_layer.DataLayer.get_input_tensors>`.
      output_values: evaluation of :meth:`self.get_output_tensors()
                                          <get_output_tensors>`.

    Returns:
      dict: dictionary with values that need to be logged to TensorBoard (can be empty).
    """
    # by default return an empty dictionary and do nothing
    return {}

  def maybe_evaluate(self, inputs_per_batch, outputs_per_batch):
    """This function can be used to calculate evaluation metrics.
    For example, for speech-to-text models this function can calculate
    word-error-rate on the validation data. For text-to-text models, this
    function can compute BLEU score. Look at the corresponding derived classes
    for examples of this. This function will be called every
    ``eval_steps`` (config parameter) iterations and
    input/output values will be populated automatically by calling ``sess.run``
    on corresponding tensors for each batch (using evaluation model). Note that
    this function is not abstract and does not have to be implemented in
    derived classes. But if evaluation functionality is required,
    overwriting this function can be a useful way to add it.

    Args:
      inputs_per_batch (list): list with evaluation of
          :meth:`self.data_layer.get_input_tensors()
          <data.data_layer.DataLayer.get_input_tensors>`
          for each batch in evaluation dataset.
      outputs_per_batch (list): list with evaluation of
          :meth:`self.get_output_tensors() <get_output_tensors>`
          for each batch in evaluation dataset.

    Returns:
      dict: dictionary with values that need to be logged to TensorBoard (can be empty).
    """
    # by default return an empty dictionary and do nothing
    return {}

  def infer(self, inputs_per_batch, outputs_per_batch, output_file):
    """ This function should be implemented if the model support inference mode.
    For example for speech-to-text and text-to-text models, this function will
    log the corresponding input-output pair to the output_file.

    Args:
      inputs_per_batch (list): list with evaluation of
          :meth:`self.data_layer.get_input_tensors()
          <data.data_layer.DataLayer.get_input_tensors>`
          for each batch in evaluation dataset.
      outputs_per_batch (list): list with evaluation of
          :meth:`self.get_output_tensors() <get_output_tensors>`
          for each batch in evaluation dataset.
      output_file (str): name of the output file that inference results should
          be saved to.
    """
    return None

  def get_output_tensors(self):
    """Returns output tensors generated by :meth:`_build_forward_pass_graph.`

    Returns:
      list: list with output tensors.
    """
    return self._outputs

  def get_tf_dtype(self):
    """Returns actual TensorFlow dtype that will be used as variables dtype."""
    if self._dtype == "mixed":
      return tf.float16
    else:
      return self._dtype

  @property
  def params(self):
    """Parameters used to construct the model (dictionary)."""
    return self._params

  @property
  def data_layer(self):
    """Model data layer."""
    return self._data_layer

  @property
  def step_size(self):
    """Number of samples the model processes per step.
    This parameter is only populated if ``num_epochs`` was specified in the
    config. It is used in training hooks to correctly print epoch number.
    """
    return self._step_size

  @property
  def last_step(self):
    """Number of steps the training should be run for."""
    return self._last_step

  @property
  def num_gpus(self):
    """Number of GPUs the model will be run on.
    For Horovod this is always 1 and actual number of GPUs is controlled by
    mpi parameters.
    """
    return self._num_gpus

  @property
  def on_horovod(self):
    """Whether the model is run on Horovod or not."""
    return self._on_horovod

  @property
  def mode(self):
    """Mode the model is executed in ("train", "eval" or "infer")."""
    return self._mode
