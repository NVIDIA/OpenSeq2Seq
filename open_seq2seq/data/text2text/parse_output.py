# Copyright (c) 2017 NVIDIA Corporation
"""
This file takes output of the inference stage produced using
TransformerDataLayer and converts it to simple tokenized text
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import argparse

import tokenizer


def main():
  with open(FLAGS.input_file, 'r') as in_file:
    def trim(token):
      return token[1:-1]

    print("******Reading from file: {}".format(FLAGS.input_file))
    with open(FLAGS.output_file, 'w') as out_file:
      print("******Writing to file: {}".format(FLAGS.output_file))
      for line in in_file:
        # merge and split by _
        escaped_tokens = "".join([trim(t) for t in line.strip().split(" ")])
        escaped_tokens = escaped_tokens.split("_")

        # unescape
        unescaped_tokens = []
        for token in escaped_tokens:
          if token:
            unescaped_tokens.append(tokenizer.unescape_token(token))

        # join and write
        out_file.write(tokenizer.join_tokens_to_string(unescaped_tokens)+'\n')
  print("******All done!")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--input_file", "-if", type=str, default="",
      help="output of the inference stage produced using model with "
           "TransformerDataLayer",
      metavar="<IF>")
  parser.add_argument(
      "--output_file", "-of", type=str, default="tokenized_output.txt",
      help="where to save output",
      metavar="<OF>")
  FLAGS, _ = parser.parse_known_args()
  main()
