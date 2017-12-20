# Copyright (c) 2017 NVIDIA Corporation
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import io
import sys
import json
import time
import tensorflow as tf
import math
from open_seq2seq.model import seq2seq_model
from open_seq2seq.data import data_layer, utils
import numpy as np

tf.flags.DEFINE_string("config_file", "",
                       """Path to the file with configuration""")
tf.flags.DEFINE_string("logdir", "",
                       """Path to where save logs and checkpoints""")
tf.flags.DEFINE_integer("summary_frequency", 20,
                       """summary step frequencey save rate""")
tf.flags.DEFINE_string("mode", "train",
                       """Mode: train - for training mode, infer - for inference mode""")
tf.flags.DEFINE_integer("bench_steps", 0,
                       """"Number of training benchmark samples""")
FLAGS = tf.flags.FLAGS

def train(config, eval_config=None):
  """
  Implements training mode
  :param config: python dictionary describing model and data layer
  :param eval_config: (default) None python dictionary describing model and data layer used for evaluation
  :return: nothing
  """
  utils.deco_print("Executing training mode")
  utils.deco_print("Creating data layer")
  dl = data_layer.ParallelDataInRamInputLayer(params=config)
  if 'pad_vocabs_to_eight' in config and config['pad_vocabs_to_eight']:
    config['src_vocab_size'] = int(math.ceil(len(dl.source_seq2idx) / 8) * 8)
    config['tgt_vocab_size'] = int(math.ceil(len(dl.target_seq2idx) / 8) * 8)
  else:
    config['src_vocab_size'] = len(dl.source_seq2idx)
    config['tgt_vocab_size'] = len(dl.target_seq2idx)
  utils.deco_print("Data layer created")

  bpe_used = False if "bpe_used" not in config else config["bpe_used"]
  gpu_ids = list(range(0, config["num_gpus"]))

  ds = list(range(0, config["num_gpus"]))

  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    sess_config = tf.ConfigProto(allow_soft_placement=True)

    # regular checkpoint saver
    saver = tf.train.Saver()

    # Create train model
    model = seq2seq_model.BasicSeq2SeqWithAttention(model_params=config,
                                                    global_step=global_step,
                                                    mode="train",
                                                    gpu_ids=gpu_ids)
    fetches_bench = [model.train_op]

    with tf.Session(config=sess_config) as sess:
      bench_steps_limit = FLAGS.bench_steps
      bench_metric = np.zeros((bench_steps_limit,2))
      bench_steps = 0
      tok_sec = np.zeros(FLAGS.summary_frequency)
      sess.run(tf.global_variables_initializer())
      utils.deco_print("Started training: Initialized variables")

      #begin training
      for epoch in range(0, config['num_epochs']):
        utils.deco_print("\n\n")
        utils.deco_print("Doing epoch {}".format(epoch))
        epoch_start = time.time()
        total_train_loss = 0.0
        t_cnt = 0
        for i, (x, y, bucket_id, len_x, len_y) in enumerate(dl.iterate_one_epoch()):
          train_step_start= time.time()
          sess.run( fetches=fetches_bench,
                    feed_dict={ model.x: x,
                                model.y: y,
                                model.x_length: len_x,
                                model.y_length: len_y
                              })
          train_step_espl = time.time() - train_step_start        
          
          # save the current step token/sec timing
          total_num_tokens = sum(len_x) + sum(len_y)
          step_tok_sec = total_num_tokens/train_step_espl
          tok_sec[ i % FLAGS.summary_frequency ] = step_tok_sec

          # process temp benchmark samples
          if i % FLAGS.summary_frequency == 0:
            if bench_steps_limit <= bench_steps:
              np.savetxt(FLAGS.logdir[:-1] + "_bench_results",bench_metric,str('%.4f'))
              print ("End of benchmark, stopping training session")
              exit(0)
            elif i > 0:
              # print the average tokens per second over the last summary period
              avg_tok_sec = np.average(tok_sec)
              std_tok_sec = np.std(tok_sec)
              utils.deco_print("\nTotal tokens/second: %f\nSTD: %f" % (avg_tok_sec, std_tok_sec))
              bench_metric[bench_steps] = avg_tok_sec, std_tok_sec
              bench_steps += 1

          t_cnt += 1

        # epoch finished
        epoch_end = time.time()
        utils.deco_print('Epoch {} training loss: {}'.format(epoch, total_train_loss / t_cnt))
        utils.deco_print("Did epoch {} in {} seconds".format(epoch, epoch_end - epoch_start))
        dl.bucketize()

      # end of epoch loop
      utils.deco_print("Saving last checkpoint")
      saver.save(sess, save_path=os.path.join(FLAGS.logdir, "model"), global_step=global_step)

def main(_):
  with open(FLAGS.config_file) as data_file:
    in_config = json.load(data_file)
  if FLAGS.mode == "train":
    utils.deco_print("Running in training mode")
    train_config = utils.configure_params(in_config, "train")
    train(train_config, None)
  else:
    raise ValueError("Unknown mode in config file")

if __name__ == "__main__":
  tf.app.run()
