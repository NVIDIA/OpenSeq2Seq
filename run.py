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
from tensorflow.core.framework import summary_pb2

tf.flags.DEFINE_string("config_file", "",
                       """Path to the file with configuration""")
tf.flags.DEFINE_string("logdir", "",
                       """Path to where save logs and checkpoints""")
tf.flags.DEFINE_string("inference_out", "stdout",
                       """where to output inference results""")
tf.flags.DEFINE_integer("checkpoint_frequency", 300,
                       """iterations after which a checkpoint is made. Only the last 5 checkpoints are saved""")
tf.flags.DEFINE_integer("summary_frequency", 20,
                       """summary step frequencey save rate""")
tf.flags.DEFINE_integer("eval_frequency", 35,
                       """iterations after which validation takes place""")
tf.flags.DEFINE_integer("max_eval_checkpoints", 5,
                        """maximum eval checkpoints to keep""")
tf.flags.DEFINE_string("mode", "train",
                       """Mode: train - for training mode, infer - for inference mode""")
tf.flags.DEFINE_float("lr", None,
                      "If not None, this will overwrite learning rate in config")

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

  eval_using_bleu = True if "eval_bleu" not in config else config["eval_bleu"]
  bpe_used = False if "bpe_used" not in config else config["bpe_used"]
  do_eval = eval_config is not None
  if do_eval:
    utils.deco_print('Creating eval data layer')
    eval_dl = data_layer.ParallelDataInRamInputLayer(params=eval_config)
    eval_use_beam_search = False if "decoder_type" not in eval_config else eval_config["decoder_type"] == "beam_search"
    if 'pad_vocabs_to_eight' in config and config['pad_vocabs_to_eight']:
      eval_config['src_vocab_size'] = int(math.ceil(len(eval_dl.source_seq2idx) / 8) * 8)
      eval_config['tgt_vocab_size'] = int(math.ceil(len(eval_dl.target_seq2idx) / 8) * 8)
    else:
      eval_config['src_vocab_size'] = len(eval_dl.source_seq2idx)
      eval_config['tgt_vocab_size'] = len(eval_dl.target_seq2idx)

  gpu_ids = list(range(0, config["num_gpus"]))

  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Create train model
    model = seq2seq_model.BasicSeq2SeqWithAttention(model_params=config,
                                                    global_step=global_step,
                                                    mode="train",
                                                    gpu_ids=gpu_ids)
    tf.summary.scalar(name="loss", tensor=model.loss)
    summary_op = tf.summary.merge_all()
    fetches = [model.loss, model.train_op, model.lr]
    fetches_s = [model.loss, model.train_op, model.final_outputs, summary_op, model.lr]

    # Create eval model. It will re-use vars from train model (force_var_reuse=True)
    if do_eval:
      e_model = seq2seq_model.BasicSeq2SeqWithAttention(model_params=eval_config,
                                                        global_step=global_step,
                                                        tgt_max_size=max(eval_config["bucket_tgt"]),
                                                        force_var_reuse=True,
                                                        mode="infer",
                                                        gpu_ids=gpu_ids[-1:])
      eval_fetches = [e_model.final_outputs]

    sess_config = tf.ConfigProto(allow_soft_placement=True)

    # regular checkpoint saver
    saver = tf.train.Saver()
    # eval checkpoint saver
    epoch_saver = tf.train.Saver(max_to_keep=FLAGS.max_eval_checkpoints)

    with tf.Session(config=sess_config) as sess:
      sw = tf.summary.FileWriter(logdir=FLAGS.logdir, graph=sess.graph, flush_secs=60)

      if tf.train.latest_checkpoint(FLAGS.logdir) is not None:
          saver.restore(sess, tf.train.latest_checkpoint(FLAGS.logdir))
          utils.deco_print("Restored checkpoint. Resuming training")
      else:
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
          # run evaluation
          if do_eval and i % FLAGS.eval_frequency == 0:
            utils.deco_print("Evaluation on validation set")
            preds = []
            targets = []
            #iterate through evaluation data
            for j, (ex, ey, ebucket_id, elen_x, elen_y) in enumerate(eval_dl.iterate_one_epoch()):
              samples = sess.run(fetches=eval_fetches,
                                 feed_dict={
                                   e_model.x: ex,
                                   e_model.x_length: elen_x,
                                 })
              samples = samples[0].predicted_ids[:, :, 0]  if eval_use_beam_search else samples[0].sample_id

              if eval_using_bleu:
                preds.extend([utils.transform_for_bleu(si,
                                             vocab=eval_dl.target_idx2seq,
                                             ignore_special=True,
                                             delim=config["delimiter"], bpe_used=bpe_used) for sample in [samples] for si in sample])
                targets.extend([[utils.transform_for_bleu(yi,
                                             vocab=eval_dl.target_idx2seq,
                                             ignore_special=True,
                                             delim=config["delimiter"], bpe_used=bpe_used)] for yii in [ey] for yi in yii])

            eval_dl.bucketize()

            if eval_using_bleu:
              eval_bleu = utils.calculate_bleu(preds, targets)
              bleu_value = summary_pb2.Summary.Value(tag="Eval_BLEU_Score", simple_value=eval_bleu)
              bleu_summary = summary_pb2.Summary(value=[bleu_value])
              sw.add_summary(summary=bleu_summary, global_step=sess.run(global_step))
              sw.flush()

            if i > 0:
              utils.deco_print("Saving EVAL checkpoint")
              epoch_saver.save(sess, save_path=os.path.join(FLAGS.logdir, "model-eval"), global_step=global_step)

          # print sample
          if i % FLAGS.summary_frequency == 0: # print arg
            loss, _, samples, sm, lr = sess.run(fetches=fetches_s,
                                        feed_dict={
                                          model.x: x,
                                          model.y: y,
                                          model.x_length: len_x,
                                          model.y_length: len_y
                                        })
            sw.add_summary(sm, global_step=sess.run(global_step))
            utils.deco_print("In epoch {}, step {} the loss is {}".format(epoch, i, loss))
            utils.deco_print("Train Source[0]:     " + utils.pretty_print_array(x[0, :],
                                                                      vocab=dl.source_idx2seq,
                                                                      delim=config["delimiter"]))
            utils.deco_print("Train Target[0]:     " + utils.pretty_print_array(y[0,:],
                                                                      vocab=dl.target_idx2seq,
                                                                      delim = config["delimiter"]))
            utils.deco_print("Train Prediction[0]: " + utils.pretty_print_array(samples.sample_id[0,:],
                                                                          vocab=dl.target_idx2seq,
                                                                          delim=config["delimiter"]))
          else:
            loss, _, lr = sess.run(fetches=fetches,
                            feed_dict={
                                model.x: x,
                                model.y: y,
                                model.x_length: len_x,
                                model.y_length: len_y
                             })
          total_train_loss += loss
          t_cnt += 1

          # save model
          if i % FLAGS.checkpoint_frequency == 0 and i > 0:  # save freq arg
            utils.deco_print("Saving checkpoint")
            saver.save(sess, save_path=os.path.join(FLAGS.logdir, "model"), global_step=global_step)

        # epoch finished
        epoch_end = time.time()
        utils.deco_print('Epoch {} training loss: {}'.format(epoch, total_train_loss / t_cnt))
        value = summary_pb2.Summary.Value(tag="TrainEpochLoss", simple_value= total_train_loss / t_cnt)
        summary = summary_pb2.Summary(value=[value])
        sw.add_summary(summary=summary, global_step=epoch)
        sw.flush()
        utils.deco_print("Did epoch {} in {} seconds".format(epoch, epoch_end - epoch_start))
        dl.bucketize()

      # end of epoch loop
        utils.deco_print("Saving last checkpoint")
      saver.save(sess, save_path=os.path.join(FLAGS.logdir, "model"), global_step=global_step)

def infer(config):
  """
  Implements inference mode
  :param config: python dictionary describing model and data layer
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
  use_beam_search = False if "decoder_type" not in config else config["decoder_type"] == "beam_search"
  utils.deco_print("Data layer created")

  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()
    model = seq2seq_model.BasicSeq2SeqWithAttention(model_params=config,
                                                    global_step=global_step,
                                                    tgt_max_size=max(config["bucket_tgt"]),
                                                    mode="infer")
    fetches = [model._final_outputs]
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    saver = tf.train.Saver()
    with tf.train.MonitoredTrainingSession(config=sess_config) as sess:
      utils.deco_print("Trying to restore from: {}".format(tf.train.latest_checkpoint(FLAGS.logdir)))
      saver.restore(sess, tf.train.latest_checkpoint(FLAGS.logdir))
      utils.deco_print("Saving inference results to: " + FLAGS.inference_out)
      if FLAGS.inference_out == "stdout":
        fout = sys.stdout
      else:
        fout = io.open(FLAGS.inference_out, 'w', encoding='utf-8')

      for i, (x, y, bucket_id, len_x, len_y) in enumerate(dl.iterate_one_epoch()):
        # need to check outputs for beam search, and if required, make a common approach
        # to handle both greedy and beam search decoding methods
        samples = sess.run(fetches=fetches,
                           feed_dict={
                               model.x: x,
                               model.x_length: len_x,
                           })
        if i % 200 == 0 and FLAGS.inference_out != "stdout":
          print(utils.pretty_print_array(samples[0].predicted_ids[:, :, 0][0] if use_beam_search else samples[0].sample_id[0],
                                         vocab=dl.target_idx2seq,
                                         ignore_special=False,
                                         delim=config["delimiter"]))
        fout.write(utils.pretty_print_array(samples[0].predicted_ids[:, :, 0][0] if use_beam_search else samples[0].sample_id[0],
                                            vocab=dl.target_idx2seq,
                                            ignore_special=True,
                                            delim=config["delimiter"]) + "\n")
      if FLAGS.inference_out != "stdout":
          fout.close()
  utils.deco_print("Inference finished")

def main(_):
  with open(FLAGS.config_file) as data_file:
    in_config = json.load(data_file)
    if FLAGS.lr is not None:
      in_config["learning_rate"] = FLAGS.lr
      utils.deco_print("using LR from command line: {}".format(FLAGS.lr))
  if FLAGS.mode == "train":
    utils.deco_print("Running in training mode")
    train_config = utils.configure_params(in_config, "train")
    if 'source_file_eval' in in_config and 'target_file_eval' in in_config:
      eval_config = utils.configure_params(in_config, "eval")
      train(train_config, eval_config)
    else:
      train(train_config, None)
  elif FLAGS.mode == "infer":
    config = utils.configure_params(in_config, "infer")
    utils.deco_print("Running in inference mode")
    infer(config)
  else:
    raise ValueError("Unknown mode in config file")

if __name__ == "__main__":
  tf.app.run()
