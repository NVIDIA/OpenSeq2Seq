import argparse
import os
import sys

from scripts import preprocess
import pandas as pd
from scripts import audio_ops

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv",required=True,type=str, help="Path of decoded transcripts csv")
parser.add_argument("--book_path", required=True, type=str, help="Path of book from librivox")
parser.add_argument("--output_path", required=True, type=str, help="Path of output folder")
args = parser.parse_args()

input_csv_filename = args.input_csv
input_csv_file = pd.read_csv(input_csv_filename)
input_book_path = args.book_path
out_file_path = args.output_path

if not os.path.isdir(out_file_path):
  os.makedirs(out_file_path)
sample_freq = 16000
book = preprocess.clean_book(input_book_path)
if not os.path.isdir(out_file_path):
  os.makedirs(out_file_path)

for idx, row in input_csv_file.iterrows():
  print("Aligning and spliting chapter {}".format(idx+1))
  audio_file_location = row["File"]
  predicted_transcript = row["Transcript"]
  start_times = row["Timestamps"].split("#")[1:-1]
  chapter = book[idx]
  aligned_data = preprocess.get_best_match(predicted_transcript, chapter)
  word_alignments = preprocess.char_to_word(aligned_data, predicted_transcript)
  sentence_alignments = preprocess.sentence_alignment(word_alignments, start_times)
  audio_file = audio_ops.AudioFile(audio_file_location, sr= sample_freq,shift=-0.4)
  csv_out = audio_file.split_into_sentences(sentence_alignments, out=True, out_file_path=out_file_path,
                                            custom_name="chapter_{}".format(idx+1))
  labels = ["Filename", "Transcript"]
  csv_df = pd.DataFrame.from_records(csv_out, columns=labels)
  csv_df.to_csv(out_file_path+"chapter_{}.csv".format(idx+1))


