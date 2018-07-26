# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import argparse
import ast
import pprint
import runpy
import copy
import os
import sys
import shutil

import tensorflow as tf
from six.moves import range
from six import string_types


from open_seq2seq.utils.utils import deco_print, flatten_dict, \
                                     nest_dict, nested_update, get_git_diff, \
                                     get_git_hash, Logger
from open_seq2seq.utils import train, infer, evaluate

from __future__ import unicode_literals
import codecs
from subword_nmt import codecs
def main(args):
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
  parser.add_argument('--enable_logs', dest='enable_logs', action='store_true',
                      help='whether to log output, git info, cmd args, etc.')
  args, unknown = parser.parse_known_args(args)

  if args.mode not in [
      'train',
      'eval',
      'train_eval',
      'infer',
      'interactive_infer'
  ]:
    raise ValueError("Mode has to be one of "
                     "['train', 'eval', 'train_eval', 'infer', "
                     "'interactive_infer']")
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
    if type(value) == int or type(value) == float or \
       isinstance(value, string_types):
      parser_unk.add_argument('--' + pm, default=value, type=type(value))
    elif type(value) == bool:
      parser_unk.add_argument('--' + pm, default=value, type=ast.literal_eval)
  config_update = parser_unk.parse_args(unknown)
  nested_update(base_config, nest_dict(vars(config_update)))

  # checking that everything is correct with log directory
  logdir = base_config['logdir']
  if args.benchmark:
    args.no_dir_check = True
  try:
    if args.enable_logs:
      ckpt_dir = os.path.join(logdir, 'logs')
    else:
      ckpt_dir = logdir
    if args.mode == 'train' or args.mode == 'train_eval':
      if os.path.isfile(logdir):
        raise IOError("There is a file with the same name as \"logdir\" "
                      "parameter. You should change the log directory path "
                      "or delete the file to continue.")

      # check if "logdir" directory exists and non-empty
      if os.path.isdir(logdir) and os.listdir(logdir) != []:
        if not args.continue_learning:
          raise IOError("Log directory is not empty. If you want to continue "
                        "learning, you should provide "
                        "\"--continue_learning\" flag")
        checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        if checkpoint is None:
          raise IOError(
              "There is no valid TensorFlow checkpoint in the "
              "{} directory. Can't load model".format(ckpt_dir)
          )
      else:
        if args.continue_learning:
          raise IOError("The log directory is empty or does not exist. "
                        "You should probably not provide "
                        "\"--continue_learning\" flag?")
        checkpoint = None
    elif (args.mode == 'infer' or args.mode == 'eval' or
        args.mode == 'interactive_infer'):
      if os.path.isdir(logdir) and os.listdir(logdir) != []:
        checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        if checkpoint is None:
          raise IOError(
              "There is no valid TensorFlow checkpoint in the "
              "{} directory. Can't load model".format(ckpt_dir)
          )
      else:
        raise IOError(
            "{} does not exist or is empty, can't restore model".format(
                ckpt_dir
            )
        )
  except IOError as e:
    if args.no_dir_check:
      print("Warning: {}".format(e))
      print("Resuming operation since no_dir_check argument was provided")
    else:
      raise

  if base_config['use_horovod']:
    if args.mode == "interactive_infer":
      raise Error("Interactive inference does not support Horovod")
    import horovod.tensorflow as hvd
    hvd.init()
    if hvd.rank() == 0:
      deco_print("Using horovod")
  else:
    hvd = None

  if args.enable_logs and args.mode != "interactive_infer":
    if hvd is None or hvd.rank() == 0:
      if not os.path.exists(logdir):
        os.makedirs(logdir)

      tm_suf = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
      shutil.copy(
          args.config_file,
          os.path.join(logdir, 'config_{}.py'.format(tm_suf)),
      )

      with open(os.path.join(logdir, 'cmd-args_{}.log'.format(tm_suf)),
                'w') as f:
        f.write(" ".join(sys.argv))

      with open(os.path.join(logdir, 'git-info_{}.log'.format(tm_suf)),
                'w') as f:
        f.write('commit hash: {}'.format(get_git_hash()))
        f.write(get_git_diff())

      old_stdout = sys.stdout
      old_stderr = sys.stderr
      stdout_log = open(
          os.path.join(logdir, 'stdout_{}.log'.format(tm_suf)), 'a', 1
      )
      stderr_log = open(
          os.path.join(logdir, 'stderr_{}.log'.format(tm_suf)), 'a', 1
      )
      sys.stdout = Logger(sys.stdout, stdout_log)
      sys.stderr = Logger(sys.stderr, stderr_log)

    base_config['logdir'] = os.path.join(logdir, 'logs')

  train_config = copy.deepcopy(base_config)
  eval_config = copy.deepcopy(base_config)
  infer_config = copy.deepcopy(base_config)

  if args.mode == 'train' or args.mode == 'train_eval':
    if 'train_params' in config_module:
      nested_update(train_config, copy.deepcopy(config_module['train_params']))
    if hvd is None or hvd.rank() == 0:
      deco_print("Training config:")
      pprint.pprint(train_config)
  if args.mode == 'eval' or args.mode == 'train_eval':
    if 'eval_params' in config_module:
      nested_update(eval_config, copy.deepcopy(config_module['eval_params']))
    if hvd is None or hvd.rank() == 0:
      deco_print("Evaluation config:")
      pprint.pprint(eval_config)
  if args.mode == "infer" or args.mode == "interactive_infer":
    if args.infer_output_file is None:
      raise ValueError("\"infer_output_file\" command line parameter is "
                       "required in inference mode")
    if args.mode == "infer" and "infer_params" in config_module:
      nested_update(infer_config, copy.deepcopy(config_module['infer_params']))
    if (args.mode == "interactive_infer"
        and "interactive_infer_params" in config_module):
      nested_update(
          infer_config,
          copy.deepcopy(config_module['interactive_infer_params'])
      )

    if hvd is None or hvd.rank() == 0:
      deco_print("Inference config:")
      pprint.pprint(infer_config)

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

    if hvd is None or hvd.rank() == 0:
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
  elif (args.mode == 'eval' or args.mode == 'infer'
      or args.mode == 'interactive_infer'):
    if hvd is None or hvd.rank() == 0:
      deco_print("Loading model from {}".format(checkpoint))

  graph = tf.Graph()
  with graph.as_default():
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
    elif args.mode == "interactive_infer":
      infer_model = base_model(
          params=infer_config,
          mode="interactive_infer",
          hvd=hvd
      )
      infer_model.compile()
      return infer_model, checkpoint, graph

  if args.enable_logs and (hvd is None or hvd.rank() == 0):
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    stdout_log.close()
    stderr_log.close()


if __name__ == '__main__':
  main(sys.argv[1:])