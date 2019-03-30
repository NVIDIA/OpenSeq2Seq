# Copyright (c) 2019 NVIDIA Corporation
"""This file takes given a logits output pickle and start and end shifts
  words to speech and writes them in a csv file

"""
from __future__ import absolute_import, division, print_function
import pickle
import argparse
import sys
import csv
import os
sys.path.append(os.getcwd())
from open_seq2seq.utils.ctc_decoder import ctc_greedy_decoder


parser = argparse.ArgumentParser(description="Infer words' timestamps from logits' dumps")
parser.add_argument("--dumpfile", required=True,
                    help="Path to the dumped logits file")
parser.add_argument("--start_shift", type=float, default=None, help="Calibration start shift")
parser.add_argument("--end_shift", type=float, default=None, help="Calibration end shift")
parser.add_argument("--calibration_file", default=None, help="Calibration parameters filepath")
parser.add_argument("--save_file", default="sample.csv")
args = parser.parse_args()
dump = pickle.load(open(args.dumpfile, "rb"))
results = dump["logits"]
vocab = dump["vocab"]
step_size = dump["step_size"]
start_shift = args.start_shift
end_shift = args.end_shift
save_file = args.save_file
calibration_file = args.calibration_file

if start_shift is None and end_shift is None:
  if calibration_file is None:
    print('Warning: no calibration parameters were provided, using zeros instead')
    start_shift, end_shift = 0, 0
  else:
    with open(calibration_file) as calib:
      line = calib.readline().split()
      start_shift = float(line[0])
      end_shift = float(line[1])

# suppose CTC blank symbol is appended to the end of vocab
blank_idx = len(vocab)

with open(save_file, "w") as csv_file:
  writer = csv.writer(csv_file, delimiter=',')
  writer.writerow(["wav_filename", "transcript", "start_time", "end_time"])

  for r in results:
    letters, starts, ends = ctc_greedy_decoder(results[r], vocab, step_size, 28, start_shift, end_shift)
    writer.writerow([r, letters,
                     ' '.join(['{:.5f}'.format(f) for f in starts]),
                     ' '.join(['{:.5f}'.format(f) for f in ends])])

  print("Results written to : {}".format(save_file))

