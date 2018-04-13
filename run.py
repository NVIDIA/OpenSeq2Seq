# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
import argparse
import pprint
import runpy
import time
import copy
import os

from open_seq2seq.utils import deco_print
from open_seq2seq.training import train, infer, evaluate
from open_seq2seq.training.model_builders import \
  create_encoder_decoder_loss_model


def main():
  parser = argparse.ArgumentParser(description='Experiment params')
  parser.add_argument("--config_file", help="Path to the configuration file")
  parser.add_argument("--mode", default='train',
                      help="Could be \"train\", \"eval\", "
                           "\"train_eval\" or \"infer\"")
  parser.add_argument("--infer_output_file",
                      help="Path to the output of inference")
  parser.add_argument('--continue_learning', dest='continue_learning',
                      action='store_true', help="whether to continue learning")
  parser.add_argument('--no_dir_check', dest='no_dir_check',
                      action='store_true',
                      help="whether to check that everything is correct "
                           "with log directory")
  parser.add_argument('--benchmark', dest='benchmark', action='store_true',
                      help='automatic config change for benchmarking')
  parser.add_argument('--bench_steps', type=int, default='20',
                      help='max_steps for benchmarking')
  args, unknown = parser.parse_known_args()

  if args.mode not in ['train', 'eval', 'train_eval', 'infer']:
    raise ValueError("Mode has to be one of "
                     "['train', 'eval', 'train_eval', 'infer']")
  config_module = runpy.run_path(args.config_file, init_globals={'tf': tf})
  base_config = config_module['base_params']

  # after we read the config, trying to overwrite some of the properties
  # with command line arguments that were passed to the script
  parser_unk = argparse.ArgumentParser()
  for pm, value in base_config.items():
    if type(value) is int or type(value) is float or type(value) is str or \
       type(value) is bool:
      parser_unk.add_argument('--' + pm, default=value, type=type(value))
  config_update = parser_unk.parse_args(unknown)
  base_config.update(vars(config_update))

  train_config = copy.deepcopy(base_config)
  eval_config = copy.deepcopy(base_config)
  infer_config = copy.deepcopy(base_config)

  if base_config['use_horovod']:
    if args.mode == "infer" or args.mode == "eval":
      raise NotImplementedError("Inference or evaluation on horovod "
                                "is not supported yet")
    if args.mode == "train_eval":
      deco_print("Evaluation during training is not yet supported on horovod, "
                 "defaulting to just doing mode=\"train\"")
      args.mode = "train"
    import horovod.tensorflow as hvd
    hvd.init()
    if hvd.rank() == 0:
      deco_print("Using horovod")
  else:
    hvd = None

  if args.mode == 'train' or args.mode == 'train_eval':
    if 'train_params' in config_module:
      train_config.update(copy.deepcopy(config_module['train_params']))
    if hvd is None or hvd.rank() == 0:
      deco_print("Training config:")
      pprint.pprint(train_config)
  if args.mode == 'eval' or args.mode == 'train_eval':
    if 'eval_params' in config_module:
      eval_config.update(copy.deepcopy(config_module['eval_params']))
    if hvd is None or hvd.rank() == 0:
      deco_print("Evaluation config:")
      pprint.pprint(eval_config)
  if args.mode == "infer":
    if args.infer_output_file is None:
      raise ValueError("\"infer_output_file\" command line parameter is "
                       "required in inference mode")
    infer_config.update(copy.deepcopy(config_module['infer_params']))
    deco_print("Inference can be run only on one GPU. Setting num_gpus to 1")
    infer_config['num_gpus'] = 1
    deco_print("Inference config:")
    pprint.pprint(infer_config)

  # checking that everything is correct with log directory
  logdir = base_config['logdir']
  if args.benchmark:
    args.no_dir_check = True
  try:
    if args.mode == 'train' or args.mode == 'train_eval':
      if os.path.isfile(logdir):
        raise IOError("There is a file with the same name as \"logdir\" "
                      "parameter. You should change the log directory path "
                      "or delete the file to continue.")

      # check if "logdir" directory exists and non-empty
      if os.path.isdir(logdir) and os.listdir(logdir) != []:
        checkpoint = tf.train.latest_checkpoint(logdir)
        if not args.continue_learning:
          raise IOError("Log directory is not empty. If you want to continue "
                        "learning, you should provide "
                        "\"--continue_learning\" flag")
        if checkpoint is None:
          raise IOError("There is no valid TensorFlow checkpoint in the "
                        "log directory. Can't restore variables.")
      else:
        if args.continue_learning:
          raise IOError("The log directory is empty or does not exist. "
                        "You should probably not provide "
                        "\"--continue_learning\" flag?")
        checkpoint = None
    elif args.mode == 'infer' or args.mode == 'eval':
      if os.path.isdir(logdir) and os.listdir(logdir) != []:
        checkpoint = tf.train.latest_checkpoint(logdir)
        if checkpoint is None:
          raise IOError("There is no valid TensorFlow checkpoint in the "
                        "{} directory. Can't load model".format(logdir))
      else:
        raise IOError(
          "{} does not exist or is empty, can't restore model".format(logdir)
        )
  except IOError as e:
    if args.no_dir_check:
      print("Warning: {}".format(e))
      print("Resuming operation since no_dir_check argument was provided")
    else:
      raise

  if args.benchmark:
    deco_print("Adjusting config for benchmarking")
    train_config['print_samples_frequency'] = None
    train_config['print_loss_frequency'] = 1
    train_config['summary_frequency'] = None
    train_config['checkpoint_frequency'] = None
    train_config['logdir'] = str("")
    if 'num_epochs' in train_config:
      del train_config['num_epochs']
    train_config['max_steps'] = args.bench_steps
    deco_print("New benchmarking config:")
    pprint.pprint(train_config)
    args.mode = "train"
    checkpoint = None

  # checking that frequencies of samples and loss are aligned
  s_fr = base_config['print_samples_frequency']
  l_fr = base_config['print_loss_frequency']
  if s_fr is not None and l_fr is not None and s_fr % l_fr != 0:
    raise ValueError("print_samples_frequency has to be a multiple of "
                     "print_loss_frequency.")

  with tf.Graph().as_default():
    # setting random seed
    rs = base_config.get('random_seed', int(time.time()))
    if hvd is not None:
      rs += hvd.rank()
    tf.set_random_seed(rs)
    np.random.seed(rs)

    if args.mode == 'train' or args.mode == 'train_eval':
      if hvd is None or hvd.rank() == 0:
        if checkpoint is None:
          deco_print("Starting training from scratch")
        else:
          deco_print(
            "Restored checkpoint from {}. Resuming training".format(checkpoint),
          )
    elif args.mode == 'eval' or args.mode == 'infer':
      deco_print("Loading model from {}".format(checkpoint))

    if args.mode == 'train':
      train_model = create_encoder_decoder_loss_model(config=train_config,
                                                      mode="train",
                                                      hvd=hvd)
      train(train_config, train_model, None, hvd=hvd)
    elif args.mode == 'train_eval':
      train_model = create_encoder_decoder_loss_model(config=train_config,
                                                      mode="train",
                                                      hvd=hvd)
      eval_model = create_encoder_decoder_loss_model(config=eval_config,
                                                     mode="eval",
                                                     hvd=hvd,
                                                     reuse=True)
      train(train_config, train_model, eval_model, hvd=hvd)
    elif args.mode == "eval":
      eval_model = create_encoder_decoder_loss_model(
        config=eval_config,
        mode="eval",
        hvd=hvd,
      )
      evaluate(eval_config, eval_model, checkpoint)
    elif args.mode == "infer":    
      infer_model = create_encoder_decoder_loss_model(config=infer_config,
                                                      mode="infer",
                                                      hvd=hvd)
      infer(infer_config, infer_model, checkpoint,
            output_file=args.infer_output_file)


if __name__ == '__main__':
  main()
