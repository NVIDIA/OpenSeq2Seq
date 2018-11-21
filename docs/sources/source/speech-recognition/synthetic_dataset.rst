.. _synthetic_data:

Creation of Synthetic Data
==========================
Creating a synthetic dataset requires a trained speech synthesis model. This tutorial assumes that you have a trained Tactron 2 with Global Style Tokens. For details on how to train this model, see :doc:`here </speech-synthesis/tacotron-2-gst>`.

We first need to create an infer csv that pairs MAILABS wav files with LibriSpeech transcripts through tacotron_gst_create_infer_csv.py. Using this infer csv, we can start generating audio through tacotron_gst_create_syn_data.py. Last, we need to create a csv that can be used to train speech recognition models that pairs the synthetic wavs with their transcripts through nsr_create_syn_train_csv.py.

Steps:
  1. Inside tacotron_gst_create_infer_csv.py, change MAILABS_data_root and libri_data_root
     to where the MAILABS and LibriSpeech datasets are located
  2. Run tacotron_gst_create_infer_csv.py, it will create a file called generate.csv
  3. Move tacotron_gst_create_syn_data.py to the top-level OpenSeq2Seq directory
     where run.py is located, or otherwise ensure that the OpenSeq2Seq imports
     will be loaded
  4. Change config_file_path, checkpoint_path, and syn_save_dir as needed. config_file_path
     should point to tacotron_gst.py. checkpoint_path should be the directory holding the
     tacotron-gst checkpoints. syn_save_dir should be an existing directory where the synthetic
     wavs will be saved to.
  5. Run tacotron_gst_create_syn_data.py. 281241 wav files will be created. Note
     that the generation of a synthetic dataset could take as long as 1 day.
  6. Repeat steps 1-5 as many times as needed. You can change the decoder dropout
     parameter if wanted on line 63 of tacotron2_decoder.py if wanted. Be sure to
     save to a different directory to avoid overwriting the files.
  7. You should have a folder containing a number of sub-directories with each
     sub-directory holding a different synthetic dataset, each the size of
     LibriSpeech, depending on how many times step 6 was done.
  8. Inside nsr_create_syn_train_csv.py, change synthetic_data_root to point on the folder
     mentioned in step 7.
  9. Run nsr_create_syn_train_csv.py, this will create synthetic_data.csv.
  10. Add synthetic_data.csv to your speech recognition config under ``dataset_files``.
      Add ``syn_enable`` and set it to True. Add ``syn_subdirs`` which is a list
      of strings containing the names of the sub-directories mentioned in step 7.
  11. Train with this new config file.

Training With Synthetic Data
============================

We found that augmenting LibriSpeech with synthetic data allowed for a large improvement in WER compared to training on only LibriSpeech and training with traditional speech augmentation. Despite the larger amount of synthetic data, models performed best if the training data was sampled at a 50-50 ratio between LibriSpeech and synthetic samples. Since the size of the synthetic dataset is the same as the training set of LibriSpeech, this should be done automatically if training using the steps above.

