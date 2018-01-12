# Copyright (c) 2017 NVIDIA Corporation
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import sys
import json
import time
import tensorflow as tf
import math
from open_seq2seq.model import seq2seq_model, model_utils
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
tf.flags.DEFINE_integer("force_num_gpus", None,
                        """If not None, will overwrite num_gpus parameter in config""")
tf.flags.DEFINE_string("mode", "train",
                       """Mode: train - for training mode, infer - for inference mode""")

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

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sw = tf.summary.FileWriter(logdir=FLAGS.logdir, flush_secs=60)
    if do_eval:
      hooks = [model_utils.EvalHook(evaluation_model=e_model, eval_dl=eval_dl, global_step=global_step,
                                    eval_frequency=FLAGS.eval_frequency, summary_writer=sw,
                                    eval_use_beam_search=eval_use_beam_search,
                                    eval_using_bleu=eval_using_bleu,bpe_used=bpe_used, delimiter=eval_config["delimiter"]),
               model_utils.SaveAtEnd(logdir=FLAGS.logdir, global_step=global_step),
               tf.train.CheckpointSaverHook(save_steps=FLAGS.checkpoint_frequency, checkpoint_dir=FLAGS.logdir)]
    else:
      hooks = [model_utils.SaveAtEnd(logdir=FLAGS.logdir, global_step=global_step),
               tf.train.CheckpointSaverHook(save_steps=FLAGS.checkpoint_frequency, checkpoint_dir=FLAGS.logdir)]
    with tf.train.MonitoredTrainingSession(checkpoint_dir = FLAGS.logdir,
                                           is_chief=True,
                                           save_summaries_steps = FLAGS.summary_frequency,
                                           config = sess_config,
                                           save_checkpoint_secs = None,
                                           log_step_count_steps = FLAGS.summary_frequency,
                                           stop_grace_period_secs = 300,
                                           hooks = hooks) as sess:
      #begin training
      for epoch in range(0, config['num_epochs']):
        utils.deco_print("\n\n")
        utils.deco_print("Doing epoch {}".format(epoch))
        epoch_start = time.time()
        total_train_loss = 0.0
        t_cnt = 0
        for i, (x, y, bucket_id, len_x, len_y) in enumerate(dl.iterate_one_epoch()):
          # print sample
          if i % FLAGS.summary_frequency == 0: # print arg
            loss, _, samples, sm, lr = sess.run(fetches=fetches_s,
                                        feed_dict={
                                          model.x: x,
                                          model.y: y,
                                          model.x_length: len_x,
                                          model.y_length: len_y
                                        })
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

        # epoch finished
        epoch_end = time.time()
        utils.deco_print('Epoch {} training loss: {}'.format(epoch, total_train_loss / t_cnt))
        value = summary_pb2.Summary.Value(tag="TrainEpochLoss", simple_value= total_train_loss / t_cnt)
        summary = summary_pb2.Summary(value=[value])
        sw.add_summary(summary=summary, global_step=epoch)
        sw.flush()
        utils.deco_print("Did epoch {} in {} seconds".format(epoch, epoch_end - epoch_start))
        dl.bucketize()

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
    if FLAGS.force_num_gpus is not None:
      utils.deco_print("Overwriting num_gpus to: %d".format(FLAGS.force_num_gpus))
      in_config['num_gpus'] = FLAGS.force_num_gpus
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
