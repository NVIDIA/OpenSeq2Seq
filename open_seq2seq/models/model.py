# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import abc
import six
import tensorflow as tf
import numpy as np
import copy
import time

from open_seq2seq.utils.utils import deco_print
from open_seq2seq.optimizers import optimize_loss, get_regularization_loss
from open_seq2seq.utils.utils import check_params
from open_seq2seq.data import MultiGPUWrapper


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
      'use_horovod': bool,
      'batch_size_per_gpu': int,
      'data_layer': None,  # could be any user defined class
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
      'learning_rate': float,
      'logdir': str,
      'num_gpus': int,  # cannot be used when gpu_ids is specified
      'gpu_ids': list,  # cannot be used when num_gpus is specified

      'save_summaries_steps': None,  # could be int or None
      'print_loss_steps': None,  # could be int or None
      'print_samples_steps': None,  # could be int or None
      'save_checkpoint_steps': None,  # could be int or None
      'eval_steps': int,

      'random_seed': int,
      'num_epochs': int,
      'max_steps': int,
      'bench_start': int,

      'data_layer_params': dict,
      'optimizer': None,  # could be class or string
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
    }

  def __init__(self, params, mode="train", hvd=None):
    """Model constructor.
    The TensorFlow graph should not be created here, but rather in the
    :meth:`self.compile() <compile>` method.

    Args:
      params (dict): parameters describing the model.
          All supported parameters are listed in :meth:`get_required_params`,
          :meth:`get_optional_params` functions.
      mode (string, optional): "train", "eval" or "infer".
          If mode is "train" all parts of the graph will be built
          (model, loss, optimizer).
          If mode is "eval", only model and loss will be built.
          If mode is "infer", only model will be built.
      hvd (optional): if Horovod is used, this should be
          ``horovod.tensorflow`` module.
          If Horovod is not used, it should be None.

    Config parameters:

      * **random_seed** (int) --- random seed to use.
      * **use_horovod** (bool) --- whether to use Horovod for distributed
        execution.
      * **num_gpus** (int) --- number of GPUs to use. This parameter cannot be
        used if ``gpu_ids`` is specified. When ``use_horovod`` is True
        this parameter is ignored.
      * **gpu_ids** (list of ints) --- GPU ids to use. This parameter cannot be
        used if ``num_gpus`` is specified. When ``use_horovod`` is True
        this parameter is ignored.
      * **batch_size_per_gpu** (int) --- batch size to use for each GPU.
      * **num_epochs** (int) --- number of epochs to run training for.
        This parameter cannot be used if ``max_steps`` is specified.
      * **max_steps** (int) --- number of steps to run training for.
        This parameter cannot be used if ``num_epochs`` is specified.
      * **save_summaries_steps** (int or None) --- how often to save summaries.
        Setting it to None disables summaries saving.
      * **print_loss_steps** (int or None) --- how often to print loss during
        training. Setting it to None disables loss printing.
      * **print_samples_steps** (int or None) --- how often to print training
        samples (input sequences, correct answers and model predictions).
        Setting it to None disables samples printing.
      * **save_checkpoint_steps** (int or None) --- how often to save model
        checkpoints. Setting it to None disables checkpoint saving.
      * **eval_steps** (int) --- how often to run evaluation during training.
        This parameter is only checked if ``--mode`` argument of ``run.py`` is
        "train\_eval". If no evaluation is needed you should use "train" mode.
      * **logdir** (string) --- path to the log directory where all checkpoints
        and summaries will be saved.
      * **data_layer** (any class derived from
        :class:`DataLayer <data.data_layer.DataLayer>`) --- data layer class
        to use.
      * **data_layer_params** (dict) --- dictionary with data layer
        configuration.
        For complete list of possible parameters see the corresponding
        class docs.
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
      * **dtype** --- model dtype. Could be either ``tf.float16``,
        ``tf.float32`` or "mixed". For details see
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
      * **automatic_loss_scaling** --- automatic loss scaling mode. Could be
        either None, "Backoff" or "Logmax". For details see
        :ref:`mixed precision training <mixed_precision>` section in docs.
      * **summaries** (list) --- which summaries to log. Could contain
        "learning_rate", "gradients", "gradient_norm", "global_gradient_norm",
        "variables", "variable_norm".
    """
    check_params(params, self.get_required_params(), self.get_optional_params())

    self._params = copy.deepcopy(params)

    # parameter checks
    self._mode = mode
    if self._mode not in ["train", "infer", "eval"]:
      raise ValueError("Mode has to be one of ['train', 'infer', 'eval']")

    if "max_steps" in params and "num_epochs" in params:
      raise ValueError("You can't provide both max_steps and num_epochs. "
                       "Please, remove one of them from the config.")
    if mode == "train":
      if "max_steps" not in params and "num_epochs" not in params:
        raise ValueError("For training mode either max_steps or "
                         "num_epochs has to be provided")

    if 'print_samples_steps' not in self._params:
      self._params['print_samples_steps'] = None
    if 'print_loss_steps' not in self._params:
      self._params['print_loss_steps'] = None
    if 'save_checkpoint_steps' not in self._params:
      self._params['save_checkpoint_steps'] = None
    if 'save_summaries_steps' not in self._params:
      self._params['save_summaries_steps'] = None

    # checking that frequencies of samples and loss are aligned
    s_fr = self._params['print_samples_steps']
    l_fr = self._params['print_loss_steps']
    if s_fr is not None and l_fr is not None and s_fr % l_fr != 0:
      raise ValueError("print_samples_steps has to be a multiple of "
                       "print_loss_steps.")

    self._hvd = hvd
    if self._hvd:
        self._gpu_ids = range(1)
    else:
      if 'gpu_ids' in self._params:
        self._gpu_ids = self._params['gpu_ids']
      elif 'num_gpus' in self._params:
        self._gpu_ids = range(self._params['num_gpus'])
      else:
        raise ValueError('Either "gpu_ids" or "num_gpus" has to '
                         'be specified in the config')

    # setting random seed
    rs = self._params.get('random_seed', int(time.time()))
    if self.on_horovod:
      rs += hvd.rank()
    tf.set_random_seed(rs)
    np.random.seed(rs)

    if 'dtype' not in self._params:
      self._params['dtype'] = tf.float32

    dl_params = self._params.get('data_layer_params', {})
    dl_params['batch_size'] = self._params['batch_size_per_gpu']
    dl_params['use_targets'] = (self._mode == "train" or self._mode == "eval")

    if self.on_horovod:
      self._data_layer = self._params['data_layer'](
        params=dl_params, model=self,
        num_workers=self._hvd.size(), worker_id=self._hvd.rank(),
      )
    else:
      dl = self._params['data_layer'](params=dl_params, model=self)
      self._data_layer = MultiGPUWrapper(dl, num_gpus=self.num_gpus)

    if self._mode == "train":
      if "max_steps" in self._params:
        self._last_step = self._params["max_steps"]
        self._step_size = None
      else:
        ds_sample_size = self._data_layer.get_size_in_samples()
        total_bs = self._data_layer.params['batch_size']
        if self.on_horovod:
          total_bs *= self._hvd.size()
        # doing a few less steps if data size is not divisible by the batch size
        self._step_size = ds_sample_size // total_bs
        self._last_step = self._params['num_epochs'] * self._step_size

    self._outputs = [None] * self.num_gpus
    self.loss = None
    self.train_op = None

  def compile(self, force_var_reuse=False):
    """TensorFlow graph is built here."""
    if 'initializer' not in self.params:
      initializer = None
    else:
      init_dict = self.params.get('initializer_params', {})
      initializer = self.params['initializer'](**init_dict)

    self.data_layer.build_graph()
    input_tensors = self.data_layer.get_input_tensors()

    if not self.on_horovod:  # not using Horovod
      # below we follow data parallelism for multi-GPU training
      losses = []
      for gpu_cnt, gpu_id in enumerate(self._gpu_ids):
        with tf.device("/gpu:{}".format(gpu_id)), tf.variable_scope(
          name_or_scope=tf.get_variable_scope(),
          # re-using variables across GPUs.
          reuse=force_var_reuse or (gpu_cnt > 0),
          initializer=initializer,
          dtype=self.get_tf_dtype(),
        ):
          deco_print("Building graph on GPU:{}".format(gpu_id))

          loss, self._outputs[gpu_cnt] = self._build_forward_pass_graph(
            [input_tensor[gpu_cnt] for input_tensor in input_tensors],
            gpu_id=gpu_cnt,
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
        deco_print(
          "Building graph in Horovod rank: {}".format(self._hvd.rank())
        )
        loss, self._outputs[0] = self._build_forward_pass_graph(input_tensors,
                                                                gpu_id=0)
        if self._mode == "train" or self._mode == "eval":
          self.loss = loss

    if self._mode == "train":
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
        on_horovod=self.on_horovod,
      )
      tf.summary.scalar(name="train_loss", tensor=self.loss)

      if not self.on_horovod or self._hvd.rank() == 0:
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
    if self.params['dtype'] == "mixed":
      return tf.float16
    else:
      return self.params['dtype']

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
    MPI parameters.
    """
    return len(self._gpu_ids)

  @property
  def mode(self):
    """Mode the model is executed in ("train", "eval" or "infer")."""
    return self._mode

  @property
  def on_horovod(self):
    """Whether the model is run on Horovod or not."""
    return self._hvd is not None
