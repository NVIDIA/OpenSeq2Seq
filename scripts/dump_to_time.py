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
print(sys.path)
from open_seq2seq.utils.ctc_decoder import ctc_greedy_decoder
args = sys.argv[1:]
parser = argparse.ArgumentParser(description='Experiment parameters')
parser.add_argument("--dumpfile", required=False, type=str, default="/raid/Speech/dump.pkl",
                    help="Path to the configuration file")
parser.add_argument("--blank_index", type=int, default=-1, help="Index of blank char")
parser.add_argument("--start_shift", type=float, default=-0.16, help="Word start shift for JASPER 10x_3 model")
parser.add_argument("--end_shift", type=float, default=0, help="Word end shift for JASPER 10x_3 model")
parser.add_argument("--save_file", type=str, default="sample.csv")
args = parser.parse_args(args)
dump = pickle.load(open(args.dumpfile, "rb"))
blank_idx = args.blank_index
results = dump["logits"]
vocab = dump["vocab"]
step_size = dump["step_size"]
start_shift = args.start_shift
end_shift = args.end_shift
save_file = args.save_file
if blank_idx == -1:
  blank_idx = len(vocab)
csv_file = open(save_file,"w")
writer = csv.writer(csv_file, delimiter=',')
writer.writerow(["File", "Transcript", "Start time", "End time"])
for r in results:
  letters, starts, ends = ctc_greedy_decoder(results[r], vocab, step_size, 28, start_shift, end_shift)
  writer.writerow([r, letters, str(starts), str(ends)])
print("Results written to : {}".format(save_file))
