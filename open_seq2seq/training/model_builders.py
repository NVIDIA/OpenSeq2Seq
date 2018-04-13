# Copyright (c) 2017 NVIDIA Corporation
"""Helper functions to create models with various topologies"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import tensorflow as tf
import copy

from open_seq2seq.models import BasicText2TextWithAttention, Speech2Text
from open_seq2seq.data import MultiGPUWrapper
from open_seq2seq.utils import check_params


def safe_fill_params(cfg_from, cfgs_to, pm_list):
  for pm in pm_list:
    for cfg_to in cfgs_to:
      if pm not in cfg_to and pm in cfg_from:
        cfg_to[pm] = copy.deepcopy(cfg_from[pm])


def create_encoder_decoder_loss_model(config, mode, hvd, reuse=False):
  """
  Creates Model Graph for Sequence to Sequence model with one Encoder,
  one Decoder and one Loss
  :param config: Model configuration
  :param mode: "train", "train_eval" or "infer"
  :param hvd: Horovod instance
  :param reuse: Whether to reuse the variables
  :return: An instance of Seq2Seq class
  """
  check_params(
    config,
    required_dict={
      'use_horovod': bool,
      'num_gpus': int,
      'summary_frequency': None,  # could be int or None
      'print_loss_frequency': None,  # could be int or None
      'print_samples_frequency': None,  # could be int or None
      'checkpoint_frequency': None,  # could be int or None
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
      'eval_frequency': int,
      'bench_start': int,
    },
  )

  safe_fill_params(
    cfg_from=config['model_params'],
    cfgs_to=[
      config['data_layer_params'],
      config['encoder_params'],
      config['decoder_params'],
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
