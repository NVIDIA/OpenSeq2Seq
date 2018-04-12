When you execute ```./create_toy_data.sh``` script, it will create the following toy data:

* Source: random sequence of numbers

* Target: reversed source sequence

The task is to learn to reverse randomly generated sequences. You can test single or multi-gpu training.

**Single GPU:**
```
$ python run.py --config_file=example_configs/toy_data_config.json --mode=train --logdir=ModelAndLogFolder
```
* You can monitor training progress with Tensorboard: 
```
$ tensorboard --logdir=ModelAndLogFolder
```


**Multi-GPU:**

We follow *data parallel* approach for training. Each GPU receives a full copy of the model and its own mini-batch of data.

To start multi-GPU training, set ```"num_gpus"``` parameter in the model config accordingly.

The ```"batch_size"``` parameter specifies batch size *per GPU*. Hence, global (or algorithmic) batch size will be equal to ```"num_gpus" * "batch_size"```

*example_configs/toy_data_config_2GPUs.json* shows how to use the multi-gpu case

```
$ python run.py --config_file=example_configs/toy_data_config_2GPUs.json --mode=train --logdir=ModelAndLogFolder
```

This will begin the training process and save the checkpoints to the ModelAndLogFolder directory

## Example inference
Once the model has been trained and saved in ``ModelAndLogFolder``, run:
```
$ python run.py --config=example_configs/toy_data_config_2GPUs.json --mode=infer --logdir=ModelAndLogFolder --inference_out=pred.txt
```

## BLEU Score Calculation using Moses Script
```
$ ./multi-bleu.perl test/toy_data/test/target.txt < pred.txt
```
If you just used the provided configs, your BLUE score should be > 98 for both single and 2 gpu runs of the toy task.