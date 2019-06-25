# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import tensorflow as tf

if hasattr(tf.compat, 'v1'):
  tf.compat.v1.disable_eager_execution()

from open_seq2seq.utils.utils import deco_print, get_base_config, create_model,\
                                     create_logdir, check_logdir, \
                                     check_base_model_logdir
from open_seq2seq.utils import train, infer, evaluate

def main():
  # Parse args and create config
  args, base_config, base_model, config_module = get_base_config(sys.argv[1:])

  if args.mode == "interactive_infer":
    raise ValueError(
        "Interactive infer is meant to be run from an IPython",
        "notebook not from run.py."
    )

#   restore_best_checkpoint = base_config.get('restore_best_checkpoint', False)
#   # Check logdir and create it if necessary
#   checkpoint = check_logdir(args, base_config, restore_best_checkpoint)

  load_model = base_config.get('load_model', None)
  restore_best_checkpoint = base_config.get('restore_best_checkpoint', False)
  base_ckpt_dir = check_base_model_logdir(load_model, args,
                                          restore_best_checkpoint)
  base_config['load_model'] = base_ckpt_dir

  # Check logdir and create it if necessary
  checkpoint = check_logdir(args, base_config, restore_best_checkpoint)

  # Initilize Horovod
  if base_config['use_horovod']:
    import horovod.tensorflow as hvd
    hvd.init()
    if hvd.rank() == 0:
      deco_print("Using horovod")
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()
  else:
    hvd = None

  if args.enable_logs:
    if hvd is None or hvd.rank() == 0:
      old_stdout, old_stderr, stdout_log, stderr_log = create_logdir(
          args,
          base_config
      )
    base_config['logdir'] = os.path.join(base_config['logdir'], 'logs')

  if args.mode == 'train' or args.mode == 'train_eval' or args.benchmark:
    if hvd is None or hvd.rank() == 0:
      if checkpoint is None or args.benchmark:
        if base_ckpt_dir:
          deco_print("Starting training from the base model")
        else:
          deco_print("Starting training from scratch")
      else:
        deco_print(
            "Restored checkpoint from {}. Resuming training".format(checkpoint),
        )
  elif args.mode == 'eval' or args.mode == 'infer':
    if hvd is None or hvd.rank() == 0:
      deco_print("Loading model from {}".format(checkpoint))

  # Create model and train/eval/infer
  with tf.Graph().as_default():
    model = create_model(
        args, base_config, config_module, base_model, hvd, checkpoint)
    hooks = None
    if ('train_params' in config_module and
        'hooks' in config_module['train_params']):
      hooks = config_module['train_params']['hooks']
    if args.mode == "train_eval":
      train(
        model[0], eval_model=model[1], debug_port=args.debug_port,
        custom_hooks=hooks)
    elif args.mode == "train":
      train(
        model, eval_model=None, debug_port=args.debug_port, custom_hooks=hooks)
    elif args.mode == "eval":
      evaluate(model, checkpoint)
    elif args.mode == "infer":
      infer(model, checkpoint, args.infer_output_file)

  if args.enable_logs and (hvd is None or hvd.rank() == 0):
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    stdout_log.close()
    stderr_log.close()


if __name__ == '__main__':
  main()
