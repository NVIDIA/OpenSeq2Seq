# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
  data_root = "/data/speech/MAILABS"
  sub_dirs = ["en_US/by_book/male/elliot_miller/hunters_space",
              "en_US/by_book/male/elliot_miller/pink_fairy_book",
              "en_US/by_book/male/elliot_miller/pirates_of_ersatz",
              "en_US/by_book/male/elliot_miller/poisoned_pen",
              "en_US/by_book/male/elliot_miller/silent_bullet",
              "en_US/by_book/female/mary_ann/northandsouth",
              "en_US/by_book/female/mary_ann/midnight_passenger",
              "en_US/by_book/female/judy_bieber/dorothy_and_wizard_oz",
              "en_US/by_book/female/judy_bieber/emerald_city_of_oz",
              "en_US/by_book/female/judy_bieber/ozma_of_oz",
              "en_US/by_book/female/judy_bieber/rinkitink_in_oz",
              "en_US/by_book/female/judy_bieber/sky_island",
              "en_US/by_book/female/judy_bieber/the_master_key",
              "en_US/by_book/female/judy_bieber/the_sea_fairies"]

  # Check to make sure all the csvs can be found
  while True:
    check = 0
    for sub_dir in sub_dirs:
      csv = os.path.join(data_root, sub_dir, "metadata.csv")
      if not os.path.isfile(csv):
        print(("{} cannot be found. Please ensure that you have"
               "entered the correct directory where you extracted the MAILABS"
               "dataset").format(csv))
        break
      else:
        check += 1
    if check == len(sub_dirs):
      break
    data_root = input("Please input where you extracted the MAILABS US dataset: ")


  # Load all csvs
  names = ["1", "2", "3"]
  _files = None
  for sub_dir in sub_dirs:
    csv = os.path.join(data_root, sub_dir, "metadata.csv")
    files = pd.read_csv(
        csv, encoding='utf-8', sep='\x7c', header=None, quoting=3, names=names)
    files['1'] = sub_dir + '/wavs/' + files['1'].astype(str)
    if _files is None:
      _files = files
    else:
      _files = _files.append(files)

  # Optionally split data into train and validation sets
  num_files = _files.shape[0]
  np.random.shuffle(_files.values)

  # Option 1: Take x% for train and 100-x % for val
  # x = 0.8
  # train, val = np.split(_files, [int(num_files/10.*x)])

  # Option 2: Take x files for val, and rest for train
  # x = 32
  # train = _files[:-x]
  # val = _files[-x:]

  # Option 3: Don't have a validation set
  train = _files
  val = None

  # Save new csvs
  train_csv = os.path.join(data_root, "train.csv")
  val_csv = os.path.join(data_root, "val.csv")

  train.to_csv(
      train_csv, encoding='utf-8', sep='\x7c',
      header=None, quoting=3, index=False)
  if val:
    val.to_csv(
        val_csv, encoding='utf-8', sep='\x7c',
        header=None, quoting=3, index=False)

  print("Change dataset_location in tacotron_gst.py to {}".format(data_root))
