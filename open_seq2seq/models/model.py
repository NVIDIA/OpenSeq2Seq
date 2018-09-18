# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import abc
import six
import tensorflow as tf
import numpy as np
import copy
import time

try:
  from inspect import signature
except ImportError:
  from funcsigs import signature

from open_seq2seq.utils.utils import deco_print, clip_last_batch
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
        'logdir': str,
        'num_gpus': int,  # cannot be used when gpu_ids is specified
        'gpu_ids': list,  # cannot be used when num_gpus is specified

        'save_summaries_steps': None,  # could be int or None
        'print_loss_steps': None,  # could be int or None
        'print_samples_steps': None,  # could be int or None
        'print_bench_info_steps': None,  # could be int or None
        'save_checkpoint_steps': None,  # could be int or None
        'restore_best_checkpoint': bool, # whether to restore best check point
        'eval_steps': int,
        'base_logdir': str,
        'finetune': bool,
        'eval_batch_size_per_gpu': int,

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
        'larc_params': dict,
        'loss_scaling': None,  # float, "Backoff" or "LogMax"
        'loss_scaling_params': dict,
        'summaries': list,
        'iter_size': int,
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
    * **print_bench_info_steps** (int or None) --- how often to print training
      benchmarking information (average number of objects processed per step).
      Setting it to None disables intermediate benchmarking printing, but
      the average information across the whole training will always be printed
      after the last iteration.
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
    * **loss_scaling** --- could be float or string. If float, static loss
      scaling is applied. If string, the corresponding automatic
      loss scaling algorithm is used. Must be one of 'Backoff'
      of 'LogMax' (case insensitive). Only used when dtype="mixed". For details
      see :ref:`mixed precision training <mixed_precision>` section in docs.
    * **loss_scaling_params** (dict) --- dictionary containing loss scaling
      parameters.
    * **summaries** (list) --- which summaries to log. Could contain
      "learning_rate", "gradients", "gradient_norm", "global_gradient_norm",
      "variables", "variable_norm", "loss_scale".
    * **iter_size** (int) --- use this parameter to emulate large batches.
      The gradients will be accumulated for ``iter_size`` number of steps before
      applying update.
    * **larc_params** --- dictionary with parameters for LARC (or LARS)
      optimization algorithms. Can contain the following parameters:

      * **larc_mode** --- Could be either "scale" (LARS) or "clip" (LARC).
        Note that it works in addition to any other optimization algorithm
        since we treat
        it as adaptive gradient clipping and learning rate adjustment.
      * **larc_eta** (float) --- LARC or LARS scaling parameter.
      * **min_update** (float) --- minimal value of the LARC (LARS) update.
      * **epsilon** (float) --- small number added to gradient norm in
        denominator for numerical stability.
    """
    check_params(params, self.get_required_params(), self.get_optional_params())

    self._params = copy.deepcopy(params)

    if self._params.get('iter_size', 1) > 1 and hvd is None:
      raise ValueError("iter_size is only supported in Horovod mode")

    # parameter checks
    self._mode = mode
    self._interactive = False
    if self._mode == "interactive_infer":
      self._mode = "infer"
      self._interactive = True

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
    if 'print_bench_info_steps' not in self._params:
      self._params['print_bench_info_steps'] = None

    self._params['finetune'] = self._params.get('finetune', False)
    self._params['base_logdir'] = self._params.get('base_logdir', None)
    self._params['eval_batch_size_per_gpu'] = self._params.get(
        'eval_batch_size_per_gpu',
        self._params['batch_size_per_gpu']
    )

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

    if self._interactive and len(self._gpu_ids) > 1:
      raise ValueError("Interactive infer is meant to be used with 1 gpu")

    # setting random seed
    rs = self._params.get('random_seed', int(time.time()))
    if self.on_horovod:
      rs += hvd.rank()
    tf.set_random_seed(rs)
    np.random.seed(rs)

    if 'dtype' not in self._params:
      self._params['dtype'] = tf.float32

    dl_params = self._params.get('data_layer_params', {})
    if mode == 'train':
      dl_params['batch_size'] = self._params['batch_size_per_gpu']
    else:
      dl_params['batch_size'] = self._params['eval_batch_size_per_gpu']
    dl_params['mode'] = self._mode
    dl_params['interactive'] = self._interactive

    if self.on_horovod:
      self._data_layer = self._params['data_layer'](
          params=dl_params, model=self,
          num_workers=self._hvd.size(), worker_id=self._hvd.rank(),
      )
    else:
      self._data_layers = []
      for worker_id in range(self.num_gpus):
        self._data_layers.append(self._params['data_layer'](
            params=dl_params, model=self,
            num_workers=self.num_gpus, worker_id=worker_id,
        ))

    if self._mode == "train":
      if "max_steps" in self._params:
        self._last_step = self._params["max_steps"]
        self._steps_in_epoch = None
      else:
        # doing a few less steps if data size is not divisible by the batch size
        self._steps_in_epoch = self.get_data_layer().get_size_in_samples() // \
                               self.get_data_layer().params['batch_size']
        if self._steps_in_epoch is None:
          raise ValueError('The data_layer is not compatible with '
                           'epoch execution, since it does not provide '
                           'get_size_in_samples() method. Either update the '
                           'data layer or switch to using "max_steps" '
                           'paremeter.')
        if self.on_horovod:
          self._steps_in_epoch //= self._hvd.size()
        else:
          self._steps_in_epoch //= self.num_gpus
        self._steps_in_epoch //= self._params.get('iter_size', 1)
        if self._steps_in_epoch == 0:
          raise ValueError("Overall batch size is too big for this dataset.")
        self._last_step = self._params['num_epochs'] * self._steps_in_epoch

    if self.on_horovod:
      self._output = None
    else:
      self._outputs = [None] * self.num_gpus

    self.loss = None
    self.train_op = None
    self.eval_losses = None
    self._num_objects_per_step = None
    self.skip_update_ph = None

  def compile(self, force_var_reuse=False, checkpoint=None, use_trt=False, precision='FP32'):
    """TensorFlow graph is built here."""
    if 'initializer' not in self.params:
      initializer = None
    else:
      init_dict = self.params.get('initializer_params', {})
      initializer = self.params['initializer'](**init_dict)

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

          if self._interactive:
            self.get_data_layer(gpu_cnt).create_interactive_placeholders()
          else:
            self.get_data_layer(gpu_cnt).build_graph()
          input_tensors = self.get_data_layer(gpu_cnt).input_tensors

          loss, self._outputs[gpu_cnt] = self.build_forward_pass_graph(
              input_tensors,
              gpu_id=gpu_cnt,
              checkpoint=checkpoint,
              use_trt=use_trt,
              precision=precision
          )
          if self._outputs[gpu_cnt] is not None and \
             not isinstance(self._outputs[gpu_cnt], list):
            raise ValueError('Decoder outputs have to be either None or list')
          if self._mode == "train" or self._mode == "eval":
            losses.append(loss)
      # end of for gpu_ind loop
      if self._mode == "train":
        self.loss = tf.reduce_mean(losses)
      if self._mode == "eval":
        self.eval_losses = losses
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
        self.get_data_layer().build_graph()
        input_tensors = self.get_data_layer().input_tensors

        loss, self._output = self._build_forward_pass_graph(input_tensors,
                                                            gpu_id=0)
        if self._output is not None and not isinstance(self._output, list):
          raise ValueError('Decoder outputs have to be either None or list')

        if self._mode == "train":
          self.loss = loss
        if self._mode == "eval":
          self.eval_losses = [loss]

    try:
      self._num_objects_per_step = [self._get_num_objects_per_step(worker_id)
                                    for worker_id in range(self.num_gpus)]
    except NotImplementedError:
      pass

    if self._mode == "train":
      if 'lr_policy' not in self.params:
        lr_policy = None
      else:
        lr_params = self.params.get('lr_policy_params', {})
        # adding default decay_steps = max_steps if lr_policy supports it and
        # different value is not provided
        func_params = signature(self.params['lr_policy']).parameters
        if 'decay_steps' in func_params and 'decay_steps' not in lr_params:
          lr_params['decay_steps'] = self._last_step
        if 'steps_per_epoch' in func_params and \
           'steps_per_epoch' not in lr_params and 'num_epochs' in self.params:
          lr_params['steps_per_epoch'] = self.steps_in_epoch
        lr_policy = lambda gs: self.params['lr_policy'](global_step=gs,
                                                        **lr_params)

      if self.params.get('iter_size', 1) > 1:
        self.skip_update_ph = tf.placeholder(tf.bool)

      self.train_op = optimize_loss(
          loss=tf.cast(self.loss, tf.float32) + get_regularization_loss(),
          dtype=self.params['dtype'],
          optimizer=self.params['optimizer'],
          optimizer_params=self.params.get('optimizer_params', {}),
          clip_gradients=self.params.get('max_grad_norm', None),
          learning_rate_decay_fn=lr_policy,
          summaries=self.params.get('summaries', None),
          larc_params=self.params.get('larc_params', None),
          loss_scaling=self.params.get('loss_scaling', 1.0),
          loss_scaling_params=self.params.get('loss_scaling_params', None),
          on_horovod=self.on_horovod,
          iter_size=self.params.get('iter_size', 1),
          skip_update_ph=self.skip_update_ph,
      )
      tf.summary.scalar(name="train_loss", tensor=self.loss)
      if self.steps_in_epoch:
        tf.summary.scalar(
            name="epoch",
            tensor=tf.floor(tf.train.get_global_step() /
                            tf.constant(self.steps_in_epoch, dtype=tf.int64)),
        )

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

  def build_forward_pass_graph(self, input_tensors, gpu_id=0, checkpoint=None, use_trt=False, precision='FP32'):
    """Wrapper around _build_forward_pass_graph with option of using TF-TRT"""
    if use_trt:
      import tensorflow.contrib.tensorrt as trt
      # Create temporary graph which will contain the native TF graph
      tf_config = tf.ConfigProto()
      tf_config.gpu_options.allow_growth = True
      temp_graph = tf.Graph()
      with temp_graph.as_default() as tf_graph:
        with tf.Session(config=tf_config) as tf_sess:
          input_placeholders = {'source_tensors': [
            tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_map1'),
            tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_map2')
            ]}
          loss, self._outputs[gpu_id] = self._build_forward_pass_graph(
              input_placeholders,
              gpu_id=gpu_id
          )
          output_node_names = [x.name.split(':0')[0] for x in self._outputs[gpu_id]]
          # Restore checkpoint here because we have to freeze the graph
          tf_saver = tf.train.Saver()
          tf_saver.restore(save_path=checkpoint, sess=tf_sess)
          frozen_graph = tf.graph_util.convert_variables_to_constants(
                tf_sess,
                tf_sess.graph_def,
                output_node_names=output_node_names
          )
          num_nodes = len(frozen_graph.node)
          print('Converting graph using TensorFlow-TensorRT...')
          frozen_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=output_node_names,
            max_batch_size=64,
            max_workspace_size_bytes=4096 << 20,
            precision_mode=precision,
            minimum_segment_size=3
          )
          print('Total node count before and after TF-TRT conversion:', num_nodes, '->', len(frozen_graph.node))
          print('TRT node count:', len([1 for n in frozen_graph.node if str(n.op)=='TRTEngineOp']))
      # Perform calibration for INT8 precision mode
      if precision == 'int8':
          with tf.Session(config=tf_config) as tf_sess:
            calib_graph = frozen_graph
            num_iterations = 10
            print('Calibrating INT8...')
            self._outputs[gpu_id] = tf.import_graph_def(calib_graph,
                input_map={'input_map1': input_tensors['source_tensors'][0]},
                return_elements=[x+':0' for x in output_node_names],
                name='')
            self._num_objects_per_step = [self._get_num_objects_per_step(worker_id)
                                    for worker_id in range(self.num_gpus)]
            results_per_batch = iterate_data(
              self, tf_sess, compute_loss=False, mode='infer', verbose=False, num_steps=num_iterations
            )
            frozen_graph = trt.calib_graph_to_infer_graph(calib_graph)
            del calib_graph
            print('INT8 graph created.')
            print('Nodes INT8:', len(frozen_graph.node))
      # Import TRT converted graph to default graph, mapping it to the original input tensors
      self._outputs[gpu_id] = tf.import_graph_def(frozen_graph,
          input_map={'input_map1': input_tensors['source_tensors'][0]},
          return_elements=[x+':0' for x in output_node_names],
          name='')
      return loss, self._outputs[gpu_id]
    else:
      return self._build_forward_pass_graph(input_tensors, gpu_id)

  @abc.abstractmethod
  def _build_forward_pass_graph(self, input_tensors, gpu_id=0):
    """Abstract method. Should create the graph of the forward pass of the model.

    Args:
      input_tensors: ``input_tensors`` defined by the data_layer class.
      gpu_id (int, optional): id of the GPU where the current copy of the model
          is constructed. For Horovod this is always zero.

    Returns:
      tuple: tuple containing loss tensor and list of outputs tensors.

      Loss tensor will be automatically provided to the optimizer and
      corresponding :attr:`train_op` will be created.

      Samples tensors are stored in the :attr:`_outputs` attribute and can be
      accessed by calling :meth:`get_output_tensors` function. For example,
      this happens inside :class:`utils.hooks.RunEvaluationHook`
      to fetch output values for evaluation.

      Both loss and outputs can be None when corresponding part of the graph
      is not built.
    """
    pass

  def maybe_print_logs(self, input_values, output_values, training_step):
    """This method can be used to print logs that help to visualize training.
    For example, you can print sample input sequences and their corresponding
    predictions. This method will be called every ``print_samples_steps``
    (config parameter) iterations and input/output values will be populated
    automatically by calling ``sess.run`` on corresponding tensors. Note that
    this method is not abstract and does not have to be implemented in
    derived classes. But if additional printing functionality is required,
    overwriting this method can be a useful way to add it.

    Args:
      input_values: evaluation of
          :meth:`self.get_data_layer(0).input_tensors
          <data.data_layer.DataLayer.input_tensors>`, that is, input tensors
          for one batch on the *first* GPU.
      output_values: evaluation of
          :meth:`self.get_output_tensors(0) <get_output_tensors>`,
          that is, output tensors for one batch on the *first* GPU.
      training_step (int): Current training step.

    Returns:
      dict: dictionary with values that need to be logged to TensorBoard
      (can be empty).
    """
    # by default return an empty dictionary and do nothing
    return {}

  def evaluate(self, input_values, output_values):
    """This method can be used in conjunction with
    :meth:`self.finalize_evaluation()<finalize_evaluation>` to calculate
    evaluation metrics.
    For example, for speech-to-text models these methods can calculate
    word-error-rate on the validation data. For text-to-text models, these
    methods can compute BLEU score. Look at the corresponding derived classes
    for examples of this. These methods will be called every
    ``eval_steps`` (config parameter) iterations and
    input/output values will be populated automatically by calling ``sess.run``
    on corresponding tensors (using evaluation model).
    The :meth:`self.evaluate()<evaluate>` method is called on each batch data
    and it's results will be collected and provided to
    :meth:`self.finalize_evaluation()<finalize_evaluation>` for finalization.
    Note that
    this function is not abstract and does not have to be implemented in
    derived classes. But if evaluation functionality is required,
    overwriting this function can be a useful way to add it.

    Args:
      input_values: evaluation of
          :meth:`self.get_data_layer().input_tensors
          <data.data_layer.DataLayer.input_tensors>` concatenated  across
          all workers. That is, input tensors for one batch combined
          from *all* GPUs.
      output_values: evaluation of
          :meth:`self.get_output_tensors() <get_output_tensors>` concatenated
          across all workers. That is, output tensors for one batch combined
          from *all* GPUs.

    Returns:
      list: all necessary values for evaluation finalization (e.g. accuracy on
      current batch, which will then be averaged in finalization method).
    """
    return []

  def finalize_evaluation(self, results_per_batch, training_step=None):
    """This method can be used in conjunction with
    :meth:`self.evaluate()<evaluate>` to calculate
    evaluation metrics.
    For example, for speech-to-text models these methods can calculate
    word-error-rate on the validation data. For text-to-text models, these
    methods can compute BLEU score. Look at the corresponding derived classes
    for examples of this. These methods will be called every
    ``eval_steps`` (config parameter) iterations and
    input/output values will be populated automatically by calling ``sess.run``
    on corresponding tensors (using evaluation model).
    The :meth:`self.evaluate()<evaluate>` method is called on each batch data
    and it's results will be collected and provided to
    :meth:`self.finalize_evaluation()<finalize_evaluation>` for finalization.
    Note that
    these methods are not abstract and does not have to be implemented in
    derived classes. But if evaluation functionality is required,
    overwriting these methods can be a useful way to add it.

    Args:
      results_per_batch (list): aggregation of values returned from all calls
          to :meth:`self.evaluate()<evaluate>` method (number of calls will be
          equal to number of evaluation batches).
      training_step (int): current training step. Will only be passed if mode
          is "train_eval".

    Returns:
      dict: dictionary with values that need to be logged to TensorBoard
      (can be empty).
    """
    # by default return an empty dictionary and do nothing
    return {}

  def infer(self, input_values, output_values):
    """This method is analogous to :meth:`self.evaluate()<evaluate>`, but used
    in conjunction with :meth:`self.finalize_inference()<finalize_inference>`
    to perform inference.

    Args:
      input_values: evaluation of
          :meth:`self.get_data_layer().input_tensors
          <data.data_layer.DataLayer.input_tensors>` concatenated  across
          all workers. That is, input tensors for one batch combined
          from *all* GPUs.
      output_values: evaluation of
          :meth:`self.get_output_tensors() <get_output_tensors>` concatenated
          across all workers. That is, output tensors for one batch combined
          from *all* GPUs.

    Returns:
      list: all necessary values for inference finalization (e.g. this method
      can return final generated sequences for each batch which will then be
      saved to file in :meth:`self.finalize_inference()<finalize_inference>`
      method).
    """
    return []

  def finalize_inference(self, results_per_batch, output_file):
    """This method should be implemented if the model support inference mode.
    For example for speech-to-text and text-to-text models, this method will
    log the corresponding input-output pair to the output_file.

    Args:
      results_per_batch (list): aggregation of values returned from all calls
          to :meth:`self.evaluate()<evaluate>` method (number of calls will be
          equal to number of evaluation batches).
      output_file (str): name of the output file that inference results should
          be saved to.
    """
    pass

  def clip_last_batch(self, last_batch, true_size):
    """This method performs last batch clipping.
    Used in cases when dataset is not divisible by the batch size and model
    does not support dynamic batch sizes. In those cases, the last batch will
    contain some data from the "next epoch" and this method can be used
    to remove that data. This method works for both
    dense and sparse tensors. In most cases you will not need to overwrite this
    method.

    Args:
      last_batch (list): list with elements that could be either ``np.array``
          or ``tf.SparseTensorValue`` containing data for last batch. The
          assumption is that the first axis of all data tensors will correspond
          to the current batch size.
      true_size (int): true size that the last batch should be cut to.
    """
    return clip_last_batch(last_batch, true_size)

  def get_output_tensors(self, worker_id=0):
    """Returns output tensors generated by :meth:`_build_forward_pass_graph.`
    When using Horovod, ``worker_id`` parameter is ignored. When using
    tower-based multi-GPU approach, ``worker_id`` can be used to select tensors
    for corresponding tower/GPU.

    Args:
      worker_id (int): id of the worker to get tensors from
          (not used for Horovod).

    Returns:
      output tensors.
    """
    if self.on_horovod:
      return self._output
    else:
      return self._outputs[worker_id]

  def get_data_layer(self, worker_id=0):
    """Returns model data layer.
    When using Horovod, ``worker_id`` parameter is ignored. When using
    tower-based multi-GPU approach, ``worker_id`` can be used to select
    data layer for corresponding tower/GPU.

    Args:
      worker_id (int): id of the worker to get data layer from
          (not used for Horovod).

    Returns:
      model data layer.
    """
    if self.on_horovod:
      return self._data_layer
    else:
      return self._data_layers[worker_id]

  def get_tf_dtype(self):
    """Returns actual TensorFlow dtype that will be used as variables dtype."""
    if self.params['dtype'] == "mixed":
      return tf.float16
    else:
      return self.params['dtype']

  def _get_num_objects_per_step(self, worker_id=0):
    """Define this method if you need benchmarking functionality.
    For example, for translation models, this method should return number of
    tokens in current batch, for image recognition model should return number
    of images in current batch.

    Args:
      worker_id (int): id of the worker to get data layer from
          (not used for Horovod).

    Returns:
      tf.Tensor with number of objects in batch.
    """
    raise NotImplementedError()

  def get_num_objects_per_step(self, worker_id=0):
    if self._num_objects_per_step:
      return self._num_objects_per_step[worker_id]
    else:
      raise NotImplementedError()

  @property
  def params(self):
    """Parameters used to construct the model (dictionary)."""
    return self._params

  @property
  def steps_in_epoch(self):
    """Number of steps in epoch.
    This parameter is only populated if ``num_epochs`` was specified in the
    config (otherwise it is None).
    It is used in training hooks to correctly print epoch number.
    """
    return self._steps_in_epoch

  @property
  def last_step(self):
    """Number of steps the training should be run for."""
    return self._last_step

  @property
  def num_gpus(self):
    """Number of GPUs the model will be run on.
    For Horovod this is always 1 and actual number of GPUs is controlled by
    Open-MPI parameters.
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

  @property
  def hvd(self):
    """horovod.tensorflow module"""
    return self._hvd
