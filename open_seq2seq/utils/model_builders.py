# Copyright (c) 2017 NVIDIA Corporation
"""Helper functions to create models with various topologies"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy

from open_seq2seq.models import BasicText2TextWithAttention, Speech2Text
from open_seq2seq.data import MultiGPUWrapper
from open_seq2seq.utils.utils import check_params


def safe_fill_params(cfg_from, cfgs_to, pm_list):
  for pm in pm_list:
    for cfg_to in cfgs_to:
      if pm not in cfg_to and pm in cfg_from:
        cfg_to[pm] = copy.deepcopy(cfg_from[pm])


def create_encoder_decoder_loss_model(config, mode, hvd, reuse=False):
  """Creates a model specified in the configuration file.
  This function takes in Python config and creates all parts of the model:
  data layer, encoder, decoder and loss. They are all then combined in one
  model instance which is returned.

  Args:
    config (dict): dictionary containing run parameters. For complete list of
        possible values see "Config parameters" section.
    mode (str): mode to create the model in. Could be "train", "eval" or "infer".
    hvd: if Horovod is used, this should be ``horovod.tensorflow`` module.
        If Horovod is not used it should be None.
    reuse (bool, optional): whether to reuse variables in the model. Useful
        for creating evaluation model during training.

  Returns:
    instance of ``base_model`` class specified in the config.

  Config parameters:

  * **random_seed** (int) --- random seed to use.
  * **use_horovod** (bool) --- whether to use Horovod for distributed execution.
  * **num_gpus** (int) --- number of GPUs to use. When ``use_horovod`` is True
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
  * **print_sample_steps** (int or None) --- how often to print training samples
    (input sequences, correct answers and model predictions).
    Setting it to None disables samples printing.
  * **save_checkpoint_steps** (int or None) --- how often to save model
    checkpoints. Setting it to None disables checkpoint saving.
  * **eval_steps** (int) --- how often to run evaluation during training.
    This parameter is only checked if ``--mode`` argument of ``run.py`` is
    "train\_eval". If no evaluation is needed you should use "train" mode.
  * **logdir** (string) --- path to the log directory where all checkpoints and
    summaries will be saved.
  * **base_model** (any class derived from
    :class:`Model <models.model.Model>`) --- base model class to use.
    Currently can only be :class:`Speech2Text <models.speech2text.Speech2Text>`
    or
    :class:`BasicText2TextWithAttention
    <models.text2text.BasicText2TextWithAttention>`.
    Note that this parameter is not a string, but an actual Python class, so you
    will need to add corresponding imports in the configuration file.
  * **model_params** (dict) --- dictionary with model configuration. For
    complete list of possible parameters see the corresponding class docs.
  * **data_layer** (any class derived from
    :class:`DataLayer <data.data_layer.DataLayer>`) --- data layer class to use.
  * **data_layer_params** (dict) --- dictionary with data layer configuration.
    For complete list of possible parameters see the corresponding class docs.
  * **encoder** (any class derived from
    :class:`Encoder <encoders.encoder.Encoder>`) --- encoder class to use.
  * **encoder_params** (dict) --- dictionary with encoder configuration. For
    complete list of possible parameters see the corresponding class docs.
  * **decoder** (any class derived from
    :class:`Decoder <decoders.decoder.Decoder>`) --- decoder class to use.
  * **decoder_params** (dict) --- dictionary with decoder configuration. For
    complete list of possible parameters see the corresponding class docs.
  * **loss** (any class derived from
    :class:`Loss <losses.loss.Loss>`) --- loss class to use.
  * **loss_params** (dict) --- dictionary with loss configuration. For
    complete list of possible parameters see the corresponding class docs.
  """
  check_params(
    config,
    required_dict={
      'use_horovod': bool,
      'num_gpus': int,
      'save_summaries_steps': None,  # could be int or None
      'print_loss_steps': None,  # could be int or None
      'print_samples_steps': None,  # could be int or None
      'save_checkpoint_steps': None,  # could be int or None
      'base_model': None,  # could be any user defined class
      'model_params': dict,
      'encoder': None,  # could be any user defined class
      'encoder_params': dict,
      'decoder': None,  # could be any user defined class
      'decoder_params': dict,
      'loss': None,  # could be any user defined class
      'loss_params': dict,
      'data_layer': None,  # could be any user defined class
      'data_layer_params': dict,
      'logdir': str,
      'batch_size_per_gpu': int,
    },
    optional_dict={
      'random_seed': int,
      'num_epochs': int,
      'max_steps': int,
      'eval_steps': int,
      'bench_start': int,
    },
  )

  safe_fill_params(
    cfg_from=config['model_params'],
    cfgs_to=[
      config['data_layer_params'],
      config['encoder_params'],
      config['decoder_params'],
      config['loss_params'],
    ],
    pm_list=['dtype'],
  )
  safe_fill_params(
    cfg_from=config,
    cfgs_to=[
      config['model_params'],
      config['encoder_params'],
      config['decoder_params'],
      config['loss_params'],
    ],
    pm_list=['batch_size_per_gpu'],
  )
  config['data_layer_params']['batch_size'] = config['batch_size_per_gpu']
  safe_fill_params(
    cfg_from=config['model_params'],
    cfgs_to=[
      config['encoder_params'],
      config['decoder_params'],
    ],
    pm_list=['regularizer', 'regularizer_params'],
  )

  if "max_steps" in config and "num_epochs" in config:
    raise ValueError("You can't provide both max_steps and num_epochs. "
                     "Please, remove one of them from the config.")
  if mode == "train":
    if "max_steps" not in config and "num_epochs" not in config:
      raise ValueError("For training mode either max_steps or "
                       "num_epochs has to be provided")

  if "max_steps" in config:
    config['model_params']['max_steps'] = config['max_steps']

  if "num_epochs" in config:
    config['model_params']['num_epochs'] = config['num_epochs']

  config['data_layer_params']['use_targets'] = (mode == "train" or
                                                mode == "eval")

  if hvd:
    data_layer = config['data_layer'](params=config['data_layer_params'])
  else:
    data_layer = MultiGPUWrapper(
      config['data_layer'](params=config['data_layer_params']),
      num_gpus=config['num_gpus'],
    )

  config['model_params']['logdir'] = config['logdir']

  if config['base_model'] == BasicText2TextWithAttention:
    config['encoder_params']['src_vocab_size'] = data_layer.params['src_vocab_size']
    config['decoder_params']['tgt_vocab_size'] = data_layer.params['tgt_vocab_size']
    config['loss_params']['tgt_vocab_size'] = data_layer.params['tgt_vocab_size']
  elif config['base_model'] == Speech2Text:
    config['decoder_params']['n_output'] = data_layer.params['alphabet'].size() + 1

  encoder = config['encoder'](params=config['encoder_params'], mode=mode)
  decoder = config['decoder'](params=config['decoder_params'], mode=mode)
  loss = config['loss'](params=config["loss_params"])

  model = config['base_model'](
    params=config['model_params'],
    data_layer=data_layer,
    encoder=encoder,
    decoder=decoder,
    loss=loss,
    mode=mode,
    force_var_reuse=reuse,
    gpu_ids=range(config['num_gpus']),
    hvd=hvd,
  )
  return model
