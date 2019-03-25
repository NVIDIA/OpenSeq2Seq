.. _speech-to-text-align:

Generating time labels for words in speech
================

######
Models
######

This approach aims at calculating the start and end time stamps of every word in a given speech sentence using the predictions from a speech recognition system. A speech recognition system (for eg.JASPER) produces a probability distribution of possible characters for each time step of the spectrogram fed into it as input. Using CTC decoder we decode it to find the best sentence equivalent for the predicted distribution.

We use the probability outputs at each time step alongwith out CTC decoder to output the start and end time for each word. However, after this we found a little shift in the predicted vs actual timestamps of the words in the sentences. This could be induced due to the convolutional layers of the network and therefore we calibrate this using reference data generated from GENTLE aligner.

######
Instructions of usage
######
1. In OpenSeq2Seq directory run::

    bash scripts/get_calibration_files.sh

This script downloads Librispeech dev-clean, extracts it and converts to 16KHz wav files. You can also change the sampling rate by replacing::

    python scripts/change_sample_rate.py --source_dir=$output_dir --target_dir=$target_dir

with::

    python scripts/change_sample_rate.py --source_dir=$output_dir --target_dir=$target_dir --sample_rate=8000

in scripts/get_calibration_files.sh

2. Run calibrate model with the same configuration as inference for your model. (make sure you have ``"infer_logits_to_pickle":True`` in the config file under decoder_params).

Example::

    python scripts/calibrate_model.py --config_file=example_configs/speech2text/jasper_10x3_10db.py

This will write a file ``calibration.txt`` with the calibration parameters.

3. For the final alignments execute as follows::

    python ./scripts/dump_to_time.py --dumpfile=<actual_model_output_pickle_file_path> --calibration_file=<path of calibration data file received in step 2>

or::

    python ./scripts/dump_to_time.py --dumpfile=<actual_model_output_pickle_file_path>  --start_shift=<start_shift_from_step2> --end_shift=<end_shift_from_step2>


This will give you a csv ``default=sample.csv`` file as follows:

.. list-table::
   :widths: 1 2 2 2
   :header-rows: 1

   * - File
     - Transcript
     - Start time
     - End time
   * - sample_file_path
     - transcript of file
     - Array with start time for each word
     - Array with end time for each word

