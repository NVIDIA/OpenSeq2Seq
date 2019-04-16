import os
import sys
import argparse
import librosa
import pandas as pd

parser = argparse.ArgumentParser(description='Conversion parameters')
parser.add_argument("--source_dir", required=False, type=str, default="calibration/sound_files/",
                    help="Path to source of flac LibriSpeech files")
parser.add_argument("--target_dir", required=False, type=str, default="calibration/sound_files_wav/",
                    help="Path to source of flac LibriSpeech files")
parser.add_argument("--sample_rate", required=False, type=int, default=16000,
                    help="Output sample rate")
parser.add_argument("--csv_out", required=False, type=bool, default=False,
                    help="Output csv file path")
args = parser.parse_args()
source_dir = args.source_dir
sample_rate = args.sample_rate
target_dir = args.target_dir

def getListOfFiles(dirName):
  """create a list of file and sub directories
  names in the given directory
  """
  listOfFile = os.listdir(dirName)
  allFiles = list()
  # Iterate over all the entries
  for entry in listOfFile:
    # Create full path
    fullPath = os.path.join(dirName, entry)
    # If entry is a directory then get the list of files in this directory
    if os.path.isdir(fullPath):
      allFiles = allFiles + getListOfFiles(fullPath)
    else:
      if fullPath[-3:] in ["wav","mp3"] or fullPath[-4:] == "flac":
        allFiles.append(fullPath)
  return allFiles


def convert_to_wav(flac_files,sample_rate,target_dir):
  """This function converts flac input to wav output of given sample rate"""
  for sound_file in flac_files:
    dir_tree = sound_file.split("/")[-4:]
    save_path = '/'.join(dir_tree[:-1])
    name = dir_tree[-1][:-4] + "wav"
    if not os.path.isdir(save_path):
      os.makedirs(save_path)
    sig, sr = librosa.load(sound_file, sample_rate)
    output_dir = target_dir+save_path
    if not os.path.isdir(output_dir):
      os.makedirs(output_dir)
    librosa.output.write_wav(output_dir + "/" + name, sig, sample_rate)

def convert(flac_files,sample_rate,target_dir):
  """This function converts flac input to wav output of given sample rate"""
  rows = []
  for sound_file in flac_files:
    dir_tree = sound_file.split("/")[-1]
    name = dir_tree.split(".")[0] + "wav"
    if not os.path.isdir(target_dir):
      os.makedirs(target_dir)
    sig, sr = librosa.load(sound_file, sample_rate)
    output_dir = target_dir
    librosa.output.write_wav(output_dir + "/" + name, sig, sample_rate)
    rows.append([output_dir + "/" + name])
  df = pd.DataFrame.from_records(data=rows, columns=["File"])
  df.to_csv(output_dir+"/chapters.csv")
input_files = getListOfFiles(source_dir)
if args.csv_out:
  convert(flac_files,sample_rate,target_dir)
else:
  convert_to_wav(input_files,sample_rate,target_dir)
