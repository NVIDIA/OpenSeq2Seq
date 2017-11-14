# Copyright (c) 2017 NVIDIA Corporation
import abc
import six
import tensorflow as tf
from .model_utils import deco_print
from .optimizers import optimize_loss

six.add_metaclass(abc.ABCMeta)
class ModelBase:
  """Abstract class that defines a sequence 2 sequence model.
  """
  def __init__(self,
               model_params,
               global_step=None,
               force_var_reuse=False,
               src_max_size=None,
               tgt_max_size=None,
               mode=None):
    """
    Constructor
    :param model_params: Python dictionary - parameters describing seq2seq model
    :param global_step: TF variable - global step
    :param force_var_reuse: Boolean - if true, all vars will be re-used
    :param src_max_size: TF tensor of shape [batch_size] - max size of src sequence
    :param tgt_max_size: TF tensor of shape [batch_size] - max size of tgt sequence
    :param mode: string, currently "train" or "infer"
    """
    self._model_params = model_params
    num_gpus = self.model_params["num_gpus"] if "num_gpus" in self.model_params else 1
    # note, that model_params["batch_size"] should specify batch_size per GPU
    # global batch size is "algo", or "global" is total batch size
    self._per_gpu_batch_size = self.model_params["batch_size"]
    self._global_batch_size = self._per_gpu_batch_size * num_gpus
    self._src_max_size = src_max_size
    self._tgt_max_size = tgt_max_size

    self._mode = self.model_params["mode"] if mode is None else mode
    if global_step is not None:
      self.global_step = global_step
    else:
      self.global_step = tf.contrib.framework.get_or_create_global_step()

    # placeholders for feeding data
    self.x = tf.placeholder(tf.int32, [self.global_batch_size, None])
    self.x_length = tf.placeholder(tf.int32, [self.global_batch_size])
    self.y = tf.placeholder(tf.int32, [self.global_batch_size, None])
    self.y_length = tf.placeholder(tf.int32, [self.global_batch_size])

    # below we follow data parallelism for multi-GPU training
    # actual per GPU data feeds
    xs = tf.split(value=self.x, num_or_size_splits=num_gpus, axis=0)
    x_lengths = tf.split(value=self.x_length, num_or_size_splits=num_gpus, axis=0)
    ys = tf.split(value=self.y, num_or_size_splits=num_gpus, axis=0)
    y_lengths = tf.split(value=self.y_length, num_or_size_splits=num_gpus, axis=0)

    losses = []

    if 'init_scale' not in self.model_params:
      initializer = None
    else:
      initializer = tf.random_uniform_initializer(-self.model_params['init_scale'], self.model_params['init_scale'])

    for gpu_ind in range(0, num_gpus):
      with tf.device("/gpu:{}".format(gpu_ind)), tf.variable_scope(
        name_or_scope=tf.get_variable_scope(),
        # re-using variables across GPUs.
        reuse=force_var_reuse or (gpu_ind > 0),
        initializer=initializer):
        deco_print("Building graph on GPU:{}".format(gpu_ind))
        if self.mode == "train":
          sample_ops, loss_i = self._build_forward_pass_graph(source_sequence = xs[gpu_ind],
                                                              src_length=x_lengths[gpu_ind],
                                                              target_sequence = ys[gpu_ind],
                                                              tgt_length=y_lengths[gpu_ind],
                                                              gpu_id=gpu_ind)
          losses.append(loss_i)

        elif self.mode == "infer":
          self._build_forward_pass_graph(source_sequence = xs[gpu_ind],
                                         src_length=x_lengths[gpu_ind],
                                         gpu_id=gpu_ind)
        else:
          raise ValueError("Unknown mode")
    # end of for gpu_ind loop

    if self.mode == "train":
      self.loss = tf.reduce_mean(losses)

    def exp_decay(learning_rate, var_global_step):
      new_lr = tf.cond(
            var_global_step < self.model_params['begin_decay_at'],
            lambda: learning_rate,
            lambda: tf.train.exponential_decay(
                learning_rate,
                var_global_step - self.model_params['begin_decay_at'],
                self.model_params['decay_steps'],
                self.model_params['decay_rate'],
                staircase=self.model_params['use_staircase_decay']),
            name="learning_rate")
      final_lr = tf.maximum(self.model_params['min_learning_rate'], new_lr)
      self._lr = final_lr
      return final_lr

    lr_decay_fn = exp_decay if 'use_decay' in self.model_params and self.model_params['use_decay'] == True else None

    if self.model_params['optimizer'].lower() == 'momentum':
      optimizer = tf.train.MomentumOptimizer(learning_rate=self.model_params['learning_rate'],
                                             momentum=0.9 if 'opt_momentum' not in self.model_params else
                                             self.model_params['opt_momentum'])
    else:
      optimizer = self.model_params['optimizer']

    if self._mode == "train":
      self._lr = tf.Variable(initial_value=self.model_params['learning_rate'], trainable=False)
      #self.train_op = tf.contrib.layers.optimize_loss(
      self.train_op = optimize_loss(
        loss = self.loss,
        global_step = tf.contrib.framework.get_global_step(),
        learning_rate = self.model_params['learning_rate'],
        optimizer = optimizer,
        gradient_noise_scale = None,
        gradient_multipliers = None,
        clip_gradients = None if 'max_grad_norm' not in self.model_params else self.model_params['max_grad_norm'],
        learning_rate_decay_fn = lr_decay_fn,
        update_ops = None,
        variables = None,
        name = "Loss_Optimization",
        summaries=["learning_rate", "loss", "gradients", "gradient_norm"],
        colocate_gradients_with_ops = True,
        increment_global_step = True,
        LARS_nu = None if 'lars_nu' not in self.model_params else self.model_params['lars_nu'],
        loss_scale = 1.0 if not "loss_scale" in self.model_params else self.model_params['loss_scale']
      )

      print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
      print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
      deco_print("Trainable variables:")
      total_params = 0
      for var in tf.trainable_variables():
        var_params = 1
        for dim in var.get_shape():
          var_params *= dim.value
        total_params += var_params
        print('Name: {}    |    Shape: {}    |    Dtype: {}'.format(var.name, tf.shape(var), var.dtype))
      deco_print('Total trainable parameters: %d' % total_params)
      print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
      print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

  @abc.abstractmethod
  def _build_forward_pass_graph(self,
                                source_sequence, src_length=None,
                                target_sequence=None, tgt_length=None,
                                gpu_id=0):
    """
    Abstract method. Should describe how forward pass graph is constructed
    :param source_sequence: 2D [batch_size, time_steps] or, potentially,
    3D [batch_size, time_steps, dim] tensor of token ids
    :param target_sequence: [batch_size, time_steps] or, potentially,
    3D [batch_size, time_steps, dim]tensor of target token ids
    :param inference_mode: Set to True for inference mode
    :param gpu id where this pass is being built
    :return: Loss tensor
    """
    pass

  @property
  def model_params(self):
    """Parameters used to construct the model"""
    return self._model_params

  @property
  def global_batch_size(self):
    """Global, or algorithmic batch size = batch_size * num_gpus """
    return self._global_batch_size

  @property
  def per_gpu_batch_size(self):
    """Per gpu_batch_size """
    return self._per_gpu_batch_size

  @property
  def src_max_size(self):
    """Maximum number of steps in source sequence"""
    return self._src_max_size

  @property
  def tgt_max_size(self):
    """Maximum number of steps in target sequence"""
    return self._tgt_max_size

  @property
  def mode(self):
    return self._mode

  @property
  def lr(self):
      return self._lr
