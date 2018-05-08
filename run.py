# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
import argparse
import pprint
import runpy
import copy
import os

from open_seq2seq.utils.utils import deco_print, flatten_dict, \
                                     nest_dict, nested_update
from open_seq2seq.utils import train, infer, evaluate


def main():
  parser = argparse.ArgumentParser(description='Experiment parameters')
  parser.add_argument("--config_file", required=True,
                      help="Path to the configuration file")
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
  parser.add_argument('--bench_start', type=int,
                      help='first step to start counting time for benchmarking')
  parser.add_argument('--debug_port', type=int,
                      help='run TensorFlow in debug mode on specified port')
  args, unknown = parser.parse_known_args()

  if args.mode not in ['train', 'eval', 'train_eval', 'infer']:
    raise ValueError("Mode has to be one of "
                     "['train', 'eval', 'train_eval', 'infer']")
  config_module = runpy.run_path(args.config_file, init_globals={'tf': tf})

  base_config = config_module.get('base_params', None)
  if base_config is None:
    raise ValueError('base_config dictionary has to be '
                     'defined in the config file')
  base_model = config_module.get('base_model', None)
  if base_model is None:
    raise ValueError('base_config class has to be defined in the config file')

  # after we read the config, trying to overwrite some of the properties
  # with command line arguments that were passed to the script
  parser_unk = argparse.ArgumentParser()
  for pm, value in flatten_dict(base_config).items():
    if type(value) is int or type(value) is float or type(value) is str or \
       type(value) is bool:
      parser_unk.add_argument('--' + pm, default=value, type=type(value))
  config_update = parser_unk.parse_args(unknown)
  nested_update(base_config, nest_dict(vars(config_update)))

  train_config = copy.deepcopy(base_config)
  eval_config = copy.deepcopy(base_config)
  infer_config = copy.deepcopy(base_config)

  if base_config['use_horovod']:
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
      # eval_config['gpu_ids'] = [eval_config['num_gpus'] - 1]
      # if 'num_gpus' in eval_config:
      #   del eval_config['num_gpus']
    if hvd is None or hvd.rank() == 0:
      deco_print("Evaluation can only be run on one GPU. "
                 "Setting num_gpus to 1 for eval model")
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
    train_config['print_samples_steps'] = None
    train_config['print_loss_steps'] = 1
    train_config['save_summaries_steps'] = None
    train_config['save_checkpoint_steps'] = None
    train_config['logdir'] = str("")
    if 'num_epochs' in train_config:
      del train_config['num_epochs']
    train_config['max_steps'] = args.bench_steps
    if args.bench_start:
      train_config['bench_start'] = args.bench_start
    elif 'bench_start' not in train_config:
      train_config['bench_start'] = 10  # default value

    deco_print("New benchmarking config:")
    pprint.pprint(train_config)
    args.mode = "train"
    checkpoint = None

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

  with tf.Graph().as_default():
    if args.mode == 'train':
      train_model = base_model(params=train_config, mode="train", hvd=hvd)
      train_model.compile()
      train(train_model, None, debug_port=args.debug_port)
    elif args.mode == 'train_eval':
      train_model = base_model(params=train_config, mode="train", hvd=hvd)
      train_model.compile()
      eval_model = base_model(params=eval_config, mode="eval", hvd=hvd)
      eval_model.compile(force_var_reuse=True)
      train(train_model, eval_model, debug_port=args.debug_port)
    elif args.mode == "eval":
      eval_model = base_model(params=eval_config, mode="eval", hvd=hvd)
      eval_model.compile()
      evaluate(eval_model, checkpoint)
    elif args.mode == "infer":
      infer_model = base_model(params=infer_config, mode="infer", hvd=hvd)
      infer_model.compile()
      infer(infer_model, checkpoint, args.infer_output_file)


if __name__ == '__main__':
  main()
