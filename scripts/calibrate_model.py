# Copyright (c) 2017 NVIDIA Corporation
"""This file helps to calculate word to speech alignments for your model
Please execute get_calibration_files.sh before running this script
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import pickle
import json
import numpy as np
import tensorflow as tf
sys.path.append(os.getcwd())

from open_seq2seq.utils.utils import deco_print, get_calibration_config, create_model,\
                                create_logdir, check_logdir, \
                                check_base_model_logdir
from open_seq2seq.utils import infer
from open_seq2seq.utils.ctc_decoder import ctc_greedy_decoder

if hasattr(tf.compat, 'v1'):
  tf.compat.v1.disable_eager_execution()


def run():
  """This function executes a saved checkpoint for
  50 LibriSpeech dev clean files whose alignments are stored in
  calibration/target.json
  This function saves a pickle file with logits after running
  through the model as calibration/sample.pkl

  :return: None
  """
  args, base_config, base_model, config_module = get_calibration_config(sys.argv[1:])
  config_module["infer_params"]["data_layer_params"]["dataset_files"] = \
    ["calibration/sample.csv"]
  config_module["base_params"]["decoder_params"]["infer_logits_to_pickle"] = True
  load_model = base_config.get('load_model', None)
  restore_best_checkpoint = base_config.get('restore_best_checkpoint',
                                            False)
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
          args, base_config
      )
      base_config['logdir'] = os.path.join(base_config['logdir'], 'logs')

  if args.mode == 'infer':
    if hvd is None or hvd.rank() == 0:
      deco_print("Loading model from {}".format(checkpoint))
  else:
    print("Run in infer mode only")
    sys.exit()
  with tf.Graph().as_default():
    model = create_model(
        args, base_config, config_module, base_model, hvd, checkpoint)
    infer(model, checkpoint, args.infer_output_file)

  return args.calibration_out


def calibrate(source, target):
  """This function calculates the mean start and end shift
  needed for your model to get word to speech alignments
  """
  print("calibrating {}".format(source))
  start_shift = []
  end_shift = []
  dump = pickle.load(open(source, "rb"))
  results = dump["logits"]
  vocab = dump["vocab"]
  step_size = dump["step_size"]
  blank_idx = len(vocab)
  with open(target, "r") as read_file:
    target = json.load(read_file)
  for wave_file in results:

    transcript, start, end = ctc_greedy_decoder(results[wave_file], vocab,
                                                step_size, blank_idx, 0, 0)
    words = transcript.split(" ")
    k = 0
    print(words)
    alignments = []
    for new_word in words:
      alignments.append({"word": new_word, "start": start[k], "end": end[k]})
      k += 1
    if len(target[wave_file]["words"]) == len(words):
      for i, new_word in enumerate(target[wave_file]["words"]):
        if new_word["case"] == "success" and \
          new_word["alignedWord"] == alignments[i]["word"]:
          start_shift.append(new_word["start"] - alignments[i]["start"])
          end_shift.append(new_word["end"] - alignments[i]["end"])
  mean_start_shift = np.mean(start_shift)
  mean_end_shift = np.mean(end_shift)
  return mean_start_shift, mean_end_shift


if __name__ == '__main__':
  calibration_out = run()
  start_mean, end_mean = calibrate("calibration/sample.pkl",
                                   "calibration/target.json")
  print("Mean start shift is {:.5f} seconds".format(start_mean))
  print("Mean end shift is: {:.5f} seconds".format(end_mean))
  with open(calibration_out, "w") as f:
    string = "{} {}".format(start_mean, end_mean)
    f.write(string)
