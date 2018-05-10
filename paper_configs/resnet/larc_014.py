from open_seq2seq.models.image2label import ResNet
from open_seq2seq.data.image2label import ImagenetDataLayer

import sys
import os
sys.path.insert(0, os.path.abspath("tensorflow-models"))
from official.resnet.resnet_run_loop import learning_rate_with_decay


batch_size_per_gpu = 32
num_gpus = 8

lr_policy_fn = learning_rate_with_decay(
  batch_size=batch_size_per_gpu * num_gpus, batch_denom=256/10.0,
  num_images=1281167, boundary_epochs=[15, 30, 40, 45],
  decay_rates=[1, 0.1, 0.01, 0.001, 1e-4],
)


def lr_policy(lr, gs, lr_policy_fn=lr_policy_fn):
  return lr_policy_fn(gs)


base_model = ResNet

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_epochs": 50,

  "num_gpus": num_gpus,
  "batch_size_per_gpu": batch_size_per_gpu,

  "save_summaries_steps": 500,
  "print_loss_steps": 10,
  "print_samples_steps": 5000,
  "eval_steps": 5000,
  "save_checkpoint_steps": 5000,
  "logdir": "experiments/resnet50-imagenet",

  "optimizer": "Momentum",
  "optimizer_params": {
    "momentum": 0.90,
  },
  "larc_params": {
    "larc_nu": 0.01,
  },
  "lr_policy": lr_policy,
  # this is ignored! LR is computed automatically from the batch size
  "learning_rate": 0.001,

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],
}

train_params = {
  "data_layer": ImagenetDataLayer,
  "data_layer_params": {
    "data_dir": "data/tf-imagenet",
  },
}

eval_params = {
  "data_layer": ImagenetDataLayer,
  "data_layer_params": {
    "data_dir": "data/tf-imagenet",
  },
}

