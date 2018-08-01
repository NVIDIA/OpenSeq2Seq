# phase 1: train the LM (for how many epochs?)
python run.py --config_file=example_configs/nlpmaster/awd-lstm.py --mode=train_eval --enable_logs

# phase 1: generate from LM (choose from best model or nah)
python run.py --config_file=example_configs/nlpmaster/awd-lstm.py --mode=infer --enable_logs

# phase 2: run on different decoders
# change checkpoint to best model from base_logdir
# save to logdir
python run.py --config_file=example_configs/nlpmaster/awd-lstm-finetune.py --mode=train_eval --enable_logs