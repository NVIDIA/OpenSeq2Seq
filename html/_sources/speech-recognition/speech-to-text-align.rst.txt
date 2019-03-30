.. _speech-to-text-align:

Generating time labels for words in speech
==========================================

######
Models
######

This approach aims at calculating the start and end time stamps of every word in a given speech sentence using the predictions from a speech recognition system. A speech recognition model (for example, Jasper) produces a probability distribution of possible characters for each time step of the spectrogram fed into it as input. Using CTC decoder we decode it to find the best sentence equivalent for the predicted distribution.

We use the probability outputs at each time step alongwith out CTC decoder to output the start and end time for each word. However, after this we compensate an offset in the predicted vs actual timestamps of the words in the sentences. This could be induced due to the convolutional layers of the network and therefore we calibrate this using reference data.

#####################
Instructions of usage
#####################
1. In OpenSeq2Seq directory run::

    bash scripts/get_calibration_files.sh

This script downloads Librispeech dev-clean, extracts it and converts to 16kHz wav files. You can also change the sampling rate by replacing::

    python scripts/change_sample_rate.py --source_dir=$output_dir --target_dir=$target_dir

with::

    python scripts/change_sample_rate.py --source_dir=$output_dir --target_dir=$target_dir --sample_rate=8000

in scripts/get_calibration_files.sh

2. Run calibrate model with your model's configuration file and checkpoint's directory.

Example::

    python scripts/calibrate_model.py --config_file=<your_model_config> --logdir=<your_model_checkpoint_directory>

This will write a file ``calibration.txt`` with the calibration parameters.

3. Get a dump with logits (predictions from your model). Please don't forget to add `'infer_logits_to_pickle': True` in the `decoder_params` section of model's configuration file. Example::

    python run.py --mode=infer --config_file=<your_model_config> --logdir=<yout_model_checkpoint_directory> --infer_output_file=<dump.pickle>

4. For the final alignments execute as follows::

    python ./scripts/dump_to_time.py --dumpfile=<actual_model_output_pickle_file_path> --calibration_file=<path of calibration data file received in step 2>

or::

    python ./scripts/dump_to_time.py --dumpfile=<actual_model_output_pickle_file_path> --start_shift=<start_shift_from_step2> --end_shift=<end_shift_from_step2>


This will give you a CSV file ``default=sample.csv`` with greedy transcripts, start time, end time per each word.
