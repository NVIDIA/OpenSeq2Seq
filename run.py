# Copyright (c) 2017 NVIDIA Corporation
import os
import sys
import json
import time
import tensorflow as tf
import copy
import math
import nltk
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

FLAGS = tf.flags.FLAGS

def train(config):
  """
  Implements training mode
  :param config: python dictionary describing model and data layer
  :return: nothing
  """
  deco_print("Executing training mode")
  deco_print("Creating data layer")
  dl = data_layer.ParallelDataInRamInputLayer(params=config)
  if 'pad_vocabs_to_eight' in config and config['pad_vocabs_to_eight']:
    config['src_vocab_size'] = math.ceil(len(dl.source_seq2idx) / 8) * 8
    config['tgt_vocab_size'] = math.ceil(len(dl.target_seq2idx) / 8) * 8
  else:
    config['src_vocab_size'] = len(dl.source_seq2idx)
    config['tgt_vocab_size'] = len(dl.target_seq2idx)
  eval_using_bleu = True if "eval_bleu" not in config else config["eval_bleu"]
  bpe_used = False if "bpe_used" not in config else config["bpe_used"]

  #create eval config
  do_eval = False
  if 'source_file_eval' in config and 'target_file_eval' in config:
    do_eval = True
    eval_config = copy.deepcopy(config)
    eval_config['source_file'] = eval_config['source_file_eval']
    eval_config['target_file'] = eval_config['target_file_eval']
    deco_print('Creating eval data layer')
    eval_dl = data_layer.ParallelDataInRamInputLayer(params=eval_config)

  deco_print("Data layer created")
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()
    #create model
    model = seq2seq_model.BasicSeq2SeqWithAttention(model_params=config,
                                                    global_step=global_step,
                                                    mode="train")
    #create eval model
    if do_eval:
      e_model = seq2seq_model.BasicSeq2SeqWithAttention(model_params=eval_config,
                                                        global_step=global_step,
                                                        force_var_reuse=True,
                                                        mode="eval")

    tf.summary.scalar(name="loss", tensor=model.loss)
    if do_eval:
      eval_fetches = [e_model._eval_y, e_model._eval_ops]
    summary_op = tf.summary.merge_all()

    fetches = [model.loss, model.train_op, model._lr]
    fetches_s = [model.loss, model.train_op, model._final_outputs, summary_op, model._lr]

    sess_config = tf.ConfigProto(allow_soft_placement=True)

    # regular checkpoint saver
    saver = tf.train.Saver()
    # eval checkpoint saver
    epoch_saver = tf.train.Saver(max_to_keep=FLAGS.max_eval_checkpoints)

    with tf.Session(config=sess_config) as sess:
      sw = tf.summary.FileWriter(logdir=FLAGS.logdir, graph=sess.graph, flush_secs=60)

      if tf.train.latest_checkpoint(FLAGS.logdir) is not None:
          saver.restore(sess, tf.train.latest_checkpoint(FLAGS.logdir))
          deco_print("Restored checkpoint. Resuming training")
      else:
          sess.run(tf.global_variables_initializer())

      #begin training
      for epoch in range(0, config['num_epochs']):
        deco_print("\n\n")
        deco_print("Doing epoch {}".format(epoch))
        epoch_start = time.time()
        total_train_loss = 0.0
        t_cnt = 0
        for i, (x, y, bucket_id, len_x, len_y) in enumerate(dl.iterate_one_epoch()):
          # run evaluation
          if do_eval and i % FLAGS.eval_frequency == 0:
            deco_print("Evaluation on validation set")
            preds = []
            targets = []
            #iterate through evaluation data
            for j, (x, y, bucket_id, len_x, len_y) in enumerate(eval_dl.iterate_one_epoch()):
              tgt, samples = sess.run(fetches=eval_fetches,
                                  feed_dict={
                                    e_model.x: x,
                                    e_model.y: y,
                                    e_model.x_length: len_x,
                                    e_model.y_length: len_y
                                  })

              if eval_using_bleu:
                preds.extend([utils.transform_for_bleu(si,
                                             vocab=eval_dl.target_idx2seq,
                                             ignore_special=True,
                                             delim=config["delimiter"], bpe_used=bpe_used) for sample in samples for si in sample.sample_id])
                targets.extend([[utils.transform_for_bleu(yi,
                                             vocab=eval_dl.target_idx2seq,
                                             ignore_special=True,
                                             delim=config["delimiter"], bpe_used=bpe_used)] for yii in tgt for yi in yii])

            eval_dl.bucketize()

            if eval_using_bleu:
              eval_bleu = calculate_bleu(preds, targets)
              bleu_value = summary_pb2.Summary.Value(tag="Eval_BLEU_Score", simple_value=eval_bleu)
              bleu_summary = summary_pb2.Summary(value=[bleu_value])
              sw.add_summary(summary=bleu_summary, global_step=sess.run(global_step))
              sw.flush()

            if i > 0:
              deco_print("Saving EVAL checkpoint")
              epoch_saver.save(sess, save_path=os.path.join(FLAGS.logdir, "model-eval"), global_step=global_step)

          # save model
          if i % FLAGS.checkpoint_frequency == 0 and i > 0: # save freq arg
              deco_print("Saving checkpoint")
              saver.save(sess, save_path=os.path.join(FLAGS.logdir, "model"), global_step=global_step)

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
            deco_print("In epoch {}, step {} the loss is {}".format(epoch, i, loss))
            deco_print("Train Source[0]:     " + utils.pretty_print_array(x[0, :],
                                                                      vocab=dl.source_idx2seq,
                                                                      delim=config["delimiter"]))
            deco_print("Train Target[0]:     " + utils.pretty_print_array(y[0,:],
                                                                      vocab=dl.target_idx2seq,
                                                                      delim = config["delimiter"]))
            deco_print("Train Prediction[0]: " + utils.pretty_print_array(samples.sample_id[0,:],
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
        deco_print('Epoch {} training loss: {}'.format(epoch, total_train_loss / t_cnt))
        value = summary_pb2.Summary.Value(tag="TrainEpochLoss", simple_value= total_train_loss / t_cnt)
        summary = summary_pb2.Summary(value=[value])
        sw.add_summary(summary=summary, global_step=epoch)
        sw.flush()
        deco_print("Did epoch {} in {} seconds".format(epoch, epoch_end - epoch_start))
        dl.bucketize()

      # end of epoch loop
      deco_print("Saving last checkpoint")
      saver.save(sess, save_path=os.path.join(FLAGS.logdir, "model"), global_step=global_step)

def infer(config):
  """
  Implements inference mode
  :param config: python dictionary describing model and data layer
  :return: nothing
  """
  deco_print("Executing training mode")
  deco_print("Creating data layer")
  dl = data_layer.ParallelDataInRamInputLayer(params=config)
  if 'pad_vocabs_to_eight' in config and config['pad_vocabs_to_eight']:
    config['src_vocab_size'] = math.ceil(len(dl.source_seq2idx) / 8) * 8
    config['tgt_vocab_size'] = math.ceil(len(dl.target_seq2idx) / 8) * 8
  else:
    config['src_vocab_size'] = len(dl.source_seq2idx)
    config['tgt_vocab_size'] = len(dl.target_seq2idx)
  use_beam_search = False if "decoder_type" not in config else config["decoder_type"] == "beam_search"
  deco_print("Data layer created")

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
      deco_print("Trying to restore from: {}".format(tf.train.latest_checkpoint(FLAGS.logdir)))
      saver.restore(sess, tf.train.latest_checkpoint(FLAGS.logdir))
      deco_print("Saving inference results to: " + FLAGS.inference_out)
      if FLAGS.inference_out == "stdout":
        fout = sys.stdout
      else:
        fout = open(FLAGS.inference_out, 'w')

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
  deco_print("Inference finished")

def calculate_bleu(preds, targets):
  '''
  :param preds: list of lists
  :param targets: list of lists
  :return: bleu score - float32
  '''
  bleu_score = nltk.translate.bleu_score.corpus_bleu(targets, preds, emulate_multibleu=True)
  print("EVAL BLEU")
  print(bleu_score)
  return bleu_score

def deco_print(line):
  print(">==================> " + line)

def configure_params(config, mode="train"):
  config["mode"] = mode
  if mode == "infer":
    config["shuffle"] = False
    config["encoder_dp_input_keep_prob"] = 1.0
    config["decoder_dp_input_keep_prob"] = 1.0
    config["batch_size"] = 1
    config["num_gpus"] = 1
    config["source_file"] = config["source_file_test"]
    config["target_file"] = config["target_file_test"]
    if "bucket_src_test" in config:
      config["bucket_src"] = config["bucket_src_test"]
    if "bucket_tgt_test" in config:
      config["bucket_tgt"] = config["bucket_tgt_test"]
  elif mode == "train":
    config["shuffle"] = True
    config["decoder_type"] = "greedy"
  else:
    raise ValueError("Unknown mode")
  return config

def main(_):
  with open(FLAGS.config_file) as data_file:
    config = json.load(data_file)
  if FLAGS.mode == "train":
    config = configure_params(config, "train")
    deco_print("Running in training mode")
    train(config)
  elif FLAGS.mode == "infer":
    config = configure_params(config, "infer")
    deco_print("Running in inference mode")
    infer(config)
  else:
    raise ValueError("Unknown mode in config file")

if __name__ == "__main__":
  tf.app.run()