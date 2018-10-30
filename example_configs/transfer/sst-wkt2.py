import tensorflow as tf

from open_seq2seq.models import LSTMLM
from open_seq2seq.encoders import LMEncoder
from open_seq2seq.decoders import FakeDecoder
from open_seq2seq.data import SSTDataLayer
from open_seq2seq.parts.rnns.weight_drop import WeightDropLayerNormBasicLSTMCell
from open_seq2seq.losses import BasicSequenceLoss, CrossEntropyLoss
from open_seq2seq.optimizers.lr_policies import fixed_lr

base_model = LSTMLM
steps = 10

data_root = "[REPLACE THIS TO THE PATH WITH YOUR SST DATA]"
processed_data_folder = 'sst-processed-data-wkt2'
binary = True
max_length = 96

base_params = {
  "restore_best_checkpoint": True, # best checkpoint is only saved when using train_eval mode
  "use_horovod": False,
  "num_gpus": 1,

  "batch_size_per_gpu": 20, 
  "eval_batch_size_per_gpu": 80,
  "num_epochs": 120,
  "save_summaries_steps": steps,
  "print_loss_steps": steps,
  "print_samples_steps": steps,
  "save_checkpoint_steps": steps,
  "load_model": "WKT2-CPT",
  "lm_vocab_file": 'wkt2-processed-data/vocab.txt',
  # "lm_vocab_file": '[LINK TO THE VOCAB FILE IN THE PROCESSED DATA USED TO TRAIN THE BASE LM]'
  "logdir": "SST-WKT2-EXP10",
  "processed_data_folder": processed_data_folder,
  "eval_steps": steps,

  "optimizer": "Adam",
  "optimizer_params": {},
  # luong10 decay scheme

  "lr_policy": fixed_lr,
  "lr_policy_params": {
    "learning_rate": 1e-4
  },

  "summaries": ['learning_rate', 'variables', 'gradients', 
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],
  # "max_grad_norm": 0.25,
  "dtype": tf.float32,
  #"dtype": "mixed",
  #"loss_scaling": "Backoff",
  "encoder": LMEncoder,
  "encoder_params": { # will need to update
    "initializer": tf.random_uniform_initializer,
    "initializer_params": { # need different initializers for embeddings and for weights
      "minval": -0.1,
      "maxval": 0.1,
    },
    "use_cudnn_rnn": False,
    "cudnn_rnn_type": None,
    "core_cell": WeightDropLayerNormBasicLSTMCell,
    "core_cell_params": {
        "num_units": 896,
        "forget_bias": 1.0,
    },
    "encoder_layers": 3,
    "encoder_dp_input_keep_prob": 1.0,
    "encoder_dp_output_keep_prob": 0.8,
    "encoder_last_input_keep_prob": 1.0,
    "encoder_last_output_keep_prob": 0.8,
    "recurrent_keep_prob": 1.0,
    'encoder_emb_keep_prob': 0.7,
    "encoder_use_skip_connections": False,
    "emb_size": 256,
    "num_tokens_gen": 10,
    "sampling_prob": 0.0, # 0 is always use the ground truth
    "fc_use_bias": True,
    "weight_tied": True,
    "awd_initializer": False,
    "use_cell_state": True,
  },

  "decoder": FakeDecoder,

  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 2e-6,
  },

  "loss": CrossEntropyLoss,
}

train_params = {
  "data_layer": SSTDataLayer,
  "data_layer_params": {
    "data_root": data_root,
    "pad_vocab_to_eight": False,
    "shuffle": True,
    "shuffle_buffer_size": 25000,
    "repeat": True,
    "map_parallel_calls": 16,
    "prefetch_buffer_size": 8,
    "max_length": max_length,
  },
}
eval_params = {
  "data_layer": SSTDataLayer,
  "data_layer_params": {
    "pad_vocab_to_eight": False,
    "shuffle": False,
    "repeat": False,
    "map_parallel_calls": 16,
    "prefetch_buffer_size": 1,
    "max_length": max_length,
  },
}

infer_params = {
  "data_layer": SSTDataLayer,
  "data_layer_params": {
    "pad_vocab_to_eight": False,
    "shuffle": False,
    "repeat": False,
    "rand_start": False,
    "map_parallel_calls": 16,
    "prefetch_buffer_size": 8,
    "max_length": max_length,
  },
}
