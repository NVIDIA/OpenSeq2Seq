# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

import tensorflow as tf

from open_seq2seq.utils.utils import deco_print, get_base_config, check_logdir,\
                                     create_logdir, create_models
from open_seq2seq.utils import train, infer, evaluate

if __name__ == '__main__':
  # Parse args and create config
  args, base_config, base_model, config_module = get_base_config(sys.argv[1:])

  if args.mode == "interactive_infer":
    deco_print(
        "WARNING: Interactive infer is meant to be run from an IPython",
        "notebook not from run.py!"
    )

  # Initilize Horovod
  if base_config['use_horovod']:
    if args.mode == "interactive_infer":
      raise ValueError("Interactive inference does not support Horovod")
    import horovod.tensorflow as hvd
    hvd.init()
    if hvd.rank() == 0:
      deco_print("Using horovod")
  else:
    hvd = None

  # Check logdir and create it if necessary
  checkpoint = check_logdir(args, base_config)
  if args.enable_logs and args.mode != "interactive_infer":
    if hvd is None or hvd.rank() == 0:
      old_stdout, old_stderr, stdout_log, stderr_log = create_logdir(
          args,
          base_config
      )
    base_config['logdir'] = os.path.join(base_config['logdir'], 'logs')

  if args.mode == 'train' or args.mode == 'train_eval' or args.benchmark:
    if hvd is None or hvd.rank() == 0:
      if checkpoint is None or args.benchmark:
        deco_print("Starting training from scratch")
      else:
        deco_print(
            "Restored checkpoint from {}. Resuming training".format(checkpoint),
        )
  elif (args.mode == 'eval' or args.mode == 'infer'
      or args.mode == 'interactive_infer'):
    if hvd is None or hvd.rank() == 0:
      deco_print("Loading model from {}".format(checkpoint))

  # Create model and train/eval/infer
  with tf.Graph().as_default():
    models = create_models(args, base_config, config_module, base_model, hvd)
    if args.mode == "train" or args.mode == "train_eval":
      train(models[0], models[1], debug_port=args.debug_port)
    elif args.mode == "eval":
      evaluate(models[1], checkpoint)
    elif args.mode == "infer":
      infer(models[2], checkpoint, args.infer_output_file)

  if args.enable_logs and (hvd is None or hvd.rank() == 0):
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    stdout_log.close()
    stderr_log.close()
