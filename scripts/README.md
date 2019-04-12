Instructions for running librivox data generator

Step 1:
Download any librivox recording book.

Step 2:
Convert audio files to 16KHz wav format.
Use the following script:
``` bash
python scripts/change_sample_rate.py --source_dir=<dir_path_of_librivox_audiofiles> --target_dir=<output_directory> --out_csv=True
```
This also creates a CSV file with path to the wav files created.


Step 3:
Run Jasper inference on these audio files

Get the latest OpenSeq2Seq master branch:
```bash
git clone https://github.com/NVIDIA/OpenSeq2Seq
```
Take any pre-trained model, add 'infer_logits_to_pickle': True to "decoder_params" section of the model's config file and put a required CSV file in "dataset_files" field of "infer_params" section.

Run inference (in order to dump logits from the model to pickle file):
```bash
python run.py --mode=infer --config="MODEL_CONFIG" --logdir="MODEL_CHECKPOINT_DIR" --num_gpus=1 --batch_size_per_gpu=1 --decoder_params/use_language_model=False --infer_output_file=model_output.pickle
```

Step 4: Install new decoder(this installs Kenlm)
```bash
chmod +x scripts/install_decoders.sh
./scripts/install_decoders.sh
```

Step 5: Build language model for each chapter of the book
```bash
python scripts/create_lm_for_chapters.py --path=<path of the book(.txt)> --wav_csv=<path of csv in created in step2> --ngrams=3 --output_path=<path where you want to store the language models>
```
This step also creates a file called <b>language_models.csv</b> in the destination folder

Step 6: Use the language models created above to create inference csv with timestamps
```bash
python scripts/infer_decoder.py --logits=<path of logits in from step 3> --labels=<path of language model csv> --alpha=4 --beta=1 --beam_width=128 --mode=lm --infer_output_file=<output_csv_file>
```

Step 7: Create and split sentences from the decoded outputs to create audio files
```bash
python generate_data_from_book.py --input_csv=<csv path from step 6> --book_path=<path of the book(.txt)> output_path=<output_dir>
```
