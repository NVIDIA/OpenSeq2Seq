import tensorflow as tf

from open_seq2seq.models import LSTMLM
from open_seq2seq.encoders import LMEncoder
from open_seq2seq.decoders import FakeDecoder
from open_seq2seq.data import WKTDataLayer
from open_seq2seq.parts.rnns.weight_drop import WeightDropLayerNormBasicLSTMCell
from open_seq2seq.losses import BasicSequenceLoss
from open_seq2seq.optimizers.lr_policies import fixed_lr

data_root = "[REPLACE THIS TO THE PATH WITH YOUR WikiText-2-raw DATA]"
processed_data_folder = 'wkt2-processed-data'

base_model = LSTMLM
bptt = 12
steps = 10

base_params = {
  "restore_best_checkpoint": True,
  "use_horovod": False,
  "num_gpus": 2,

  "batch_size_per_gpu": 160,
  "num_epochs": 1500,
  "save_summaries_steps": steps,
  "print_loss_steps": steps,
  "print_samples_steps": steps,
  "save_checkpoint_steps": steps,
  "processed_data_folder": processed_data_folder,
  "logdir": "LSTM-FP32-2GPU-SMALL",
  "eval_steps": steps * 2,

  "optimizer": "Adam",
  "optimizer_params": {},

  "lr_policy": fixed_lr,
  "lr_policy_params": {
    "learning_rate": 9e-4
  },

  "summaries": ['learning_rate', 'variables', 'gradients', 
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],

  "dtype": tf.float32,
  # "dtype": "mixed",
  # "loss_scaling": "Backoff",
  "encoder": LMEncoder,
  "encoder_params": {
    "initializer": tf.random_uniform_initializer,
    "initializer_params": {
      "minval": -0.1,
      "maxval": 0.1,
    },
    "use_cudnn_rnn": True,
    "cudnn_rnn_type": tf.contrib.cudnn_rnn.CudnnLSTM,
    "core_cell": None,
    "core_cell_params": {
        "num_units": 128,
        "forget_bias": 1.0,
    },
    "encoder_layers": 2,
    "encoder_dp_input_keep_prob": 1.0,
    "encoder_dp_output_keep_prob": 0.6,
    "encoder_last_input_keep_prob": 1.0,
    "encoder_last_output_keep_prob": 0.6,
    "recurrent_keep_prob": 0.7,
    'encoder_emb_keep_prob': 0.37,
    "encoder_use_skip_connections": False,
    "emb_size": 64,
    "sampling_prob": 0.0, # 0 is always use the ground truth
    "fc_use_bias": True,
    "weight_tied": True,
    "awd_initializer": False,
  },

  "decoder": FakeDecoder,

  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 2e-6,
  },

  "loss": BasicSequenceLoss,
  "loss_params": {
    "offset_target_by_one": False,
    "average_across_timestep": True,
    "do_mask": False,
  }
}

train_params = {
  "data_layer": WKTDataLayer,
  "data_layer_params": {
    "data_root": data_root,
    "processed_data_folder": processed_data_folder,
    "pad_vocab_to_eight": False,
    "rand_start": True,
    "shuffle": False,
    "shuffle_buffer_size": 25000,
    "repeat": True,
    "map_parallel_calls": 16,
    "prefetch_buffer_size": 8,
    "bptt": bptt,
    "small": True,
  },
}
eval_params = {
  "data_layer": WKTDataLayer,
  "data_layer_params": {
    "processed_data_folder": processed_data_folder,
    "pad_vocab_to_eight": False,
    "shuffle": False,
    "repeat": False,
    "map_parallel_calls": 16,
    "prefetch_buffer_size": 1,
    "bptt": bptt,
    "small": True,
  },
}

infer_params = {
  "data_layer": WKTDataLayer,
  "data_layer_params": {
    "processed_data_folder": processed_data_folder,
    "pad_vocab_to_eight": False,
    "shuffle": False,
    "repeat": False,
    "rand_start": False,
    "map_parallel_calls": 16,
    "prefetch_buffer_size": 8,
    "bptt": bptt,
    "seed_tokens": "something The only game",
  },
}
