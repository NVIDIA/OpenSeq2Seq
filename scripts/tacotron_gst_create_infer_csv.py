# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
  MAILABS_data_root = "/data/speech/MAILABS"
  libri_data_root = "/data/speech/librispeech"

  libri_csvs = ["librivox-train-clean-100.csv",
                "librivox-train-clean-360.csv",
                "librivox-train-other-500.csv"]
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
      csv = os.path.join(MAILABS_data_root, sub_dir, "metadata.csv")
      if not os.path.isfile(csv):
        print(("{} cannot be found. Please ensure that you have"
               "entered the correct directory where you extracted the MAILABS"
               "dataset").format(csv))
        break
      else:
        check += 1
    if check == len(sub_dirs):
      break
    MAILABS_data_root = input("Please input where you extracted the MAILABS US"
                              " dataset: ")

  while True:
    check = 0
    for csv_file in libri_csvs:
      csv = os.path.join(libri_data_root, csv_file)
      if not os.path.isfile(csv):
        print(("{} cannot be found. Please ensure that you have"
               "entered the correct directory where you extracted the"
               "librispeech dataset").format(csv))
        break
      else:
        check += 1
    if check == len(libri_csvs):
      break
    libri_data_root = input("Please input where you extracted the librispeech"
                            " dataset: ")


  # Load libri csvs
  libri_files = None
  for csv in libri_csvs:
    csv = os.path.join(libri_data_root, csv)
    file = pd.read_csv(csv, encoding='utf-8', quoting=3)
    if libri_files is None:
      libri_files = file
    else:
      libri_files = libri_files.append(file)

  # Load MAILABS csvs
  MAILABS_files = None
  names = ["1", "2", "3"]
  for sub_dir in sub_dirs:
    csv = os.path.join(MAILABS_data_root, sub_dir, "metadata.csv")
    files = pd.read_csv(
        csv, encoding='utf-8', sep='\x7c', header=None, quoting=3, names=names)
    files['1'] = sub_dir + '/wavs/' + files['1'].astype(str)
    if MAILABS_files is None:
      MAILABS_files = files
    else:
      MAILABS_files = MAILABS_files.append(files)

  num_M_files = MAILABS_files.shape[0]
  np.random.shuffle(MAILABS_files.values)

  curr_M_i = 0
  num_libri = libri_files.shape[0]

  # Mix MAILABS wavs with libri transcripts
  for i, row in enumerate(libri_files.itertuples()):
    libri_files.iat[i, 0] = MAILABS_files.iloc[curr_M_i, 0]
    libri_files.iat[i, 1] = -1
    curr_M_i += 1
    if curr_M_i >= num_M_files:
      curr_M_i = 0
    if i % int(num_libri/100) == 0:
      print("Processed {} out of {}".format(i, num_libri))
  libri_files.to_csv(
      "generate.csv", encoding='utf-8', sep='\x7c',
      header=None, quoting=3, index=False)
