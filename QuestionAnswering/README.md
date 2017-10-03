# Intro

This research project implements a variation of R-NET network [1] for question answering task.

# Speedup R-NET with CudnnLSTM [2] and Attention Modifications
| MODEL          | TRAINING TIME (50 EPOCHS) | F1 ON DEV | EM ON DEV | CONFIG    |
|:--------------:|:-------------------------:|:---------:|:---------:|:---------:|
| baseline       |~ day          | 69.67     | 59.33     | SQuAD_encoder_layer_2_encoder_size_32_layer_1_baseline_gpu_2.json |
| cuDNN_LSTMCell |~ day (1.1x)      | 69.15     | 58.78     | SQuAD_encoder_layer_2_encoder_size_32_layer_1_cudnn_rnn_cell_gpu_2.json |
| CudnnLSTM      |~ few hours (5.9x)         | 66.77     | 56.40     | SQuAD_encoder_layer_2_encoder_size_32_layer_1_cudnn_rnn_gpu_2.json |

### Requirements
  * TensorFlow v1.3+
  * NLTK v3.2.4+

### Installation

```
$ sh setup.sh
```

### Data processing (included in setup.sh)

```
$ python3 data_preprocessing.py --file datasets/SQuAD/train-v1.1.json --mode train
$ python3 data_preprocessing.py --file datasets/SQuAD/dev-v1.1.json --mode dev
```

### rNet Model
  * train

```
$ python3 run.py --config=configs/SQuAD_encoder_layer_2_encoder_size_32_layer_1_cudnn_rnn_gpu_2.json --logdir=output
```

  * infer

```
$ python3 run.py --config=configs/SQuAD_encoder_layer_2_encoder_size_32_layer_1_cudnn_rnn_infer.json --logdir=output --inference_out=output/inference_out.txt
$ python3 generate_candidates.py --file_id datasets/SQuAD/dev/id.txt --source datasets/SQuAD/dev/src/sources.txt --target output/inference_out.txt
```

  * eval

```
$ python3 eval_SQuAD.py datasets/SQuAD/dev-v1.1.json output/candidates.json
```

### Authors
[Luyang Chen](https://github.com/LouisChen1992) (internship project @ NVIDIA)

## References
1. [R-NET: MACHINE READING COMPREHENSION WITH SELF-MATCHING NETWORKS](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)
2. [Optimizing Recurrent Neural Networks in cuDNN 5](https://devblogs.nvidia.com/parallelforall/optimizing-recurrent-neural-networks-cudnn-5/)
