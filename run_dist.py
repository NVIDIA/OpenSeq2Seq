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
import horovod.tensorflow as hvd

tf.flags.DEFINE_string("config_file", "",
                       """Path to the file with configuration""")
tf.flags.DEFINE_string("logdir", "",
                       """Path to where save logs and checkpoints""")
tf.flags.DEFINE_string("inference_out", "stdout",
                       """where to output inference results""")
tf.flags.DEFINE_integer("checkpoint_frequency", 60,
                       """How often (in seconds) to save checkpoints""")
tf.flags.DEFINE_integer("summary_frequency", 20,
                       """summary step frequencey save rate""")
tf.flags.DEFINE_integer("eval_frequency", 35,
                       """iterations after which validation takes place""")
tf.flags.DEFINE_integer("max_eval_checkpoints", 5,
                        """maximum eval checkpoints to keep""")
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
  hvd.init()
  utils.deco_print("Executing training mode")
  utils.deco_print("Creating data layer")
  dl = data_layer.ParallelDataInRamInputLayer(params = config,
                                              num_workers = hvd.size(),
                                              worker_id = hvd.rank())
  if 'pad_vocabs_to_eight' in config and config['pad_vocabs_to_eight']:
    config['src_vocab_size'] = int(math.ceil(len(dl.source_seq2idx) / 8) * 8)
    config['tgt_vocab_size'] = int(math.ceil(len(dl.target_seq2idx) / 8) * 8)
  else:
    config['src_vocab_size'] = len(dl.source_seq2idx)
    config['tgt_vocab_size'] = len(dl.target_seq2idx)
  utils.deco_print("Data layer created")

  eval_using_bleu = True if "eval_bleu" not in config else config["eval_bleu"]
  bpe_used = False if "bpe_used" not in config else config["bpe_used"]
  do_eval = eval_config is not None and hvd.rank() == 0#hvd.size() - 1
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

  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Create train model
    model = seq2seq_model.BasicSeq2SeqWithAttention(model_params=config,
                                                    global_step=global_step,
                                                    mode="train",
                                                    gpu_ids="horovod")

    if do_eval:
      e_model = seq2seq_model.BasicSeq2SeqWithAttention(model_params=eval_config,
                                                        global_step=global_step,
                                                        tgt_max_size=max(eval_config["bucket_tgt"]),
                                                        force_var_reuse=True,
                                                        mode="infer",
                                                        gpu_ids=[0])
      eval_fetches = [e_model.final_outputs]

    tf.summary.scalar(name="loss", tensor=model.loss)
    summary_op = tf.summary.merge_all()
    fetches = [model.loss, model.train_op, model.lr]
    if hvd.rank()==0:
      fetches_s = [model.loss, model.train_op, model.final_outputs, summary_op, model.lr]

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    checkpoint_dir = FLAGS.logdir if hvd.rank() == 0 else None

    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           save_summaries_steps=FLAGS.summary_frequency,
                                           config=sess_config,
                                           save_checkpoint_secs=FLAGS.checkpoint_frequency,
                                           hooks=hooks) as sess:
      if hvd.rank() == 0:
        sw = tf.summary.FileWriter(logdir=FLAGS.logdir, flush_secs=60)
        #eval_saver = tf.train.Saver(max_to_keep=FLAGS.max_eval_checkpoints)
      #begin training
      for epoch in range(0, config['num_epochs']):
        utils.deco_print("\n\n")
        utils.deco_print("Doing epoch {} on Horovod rank {}".format(epoch, hvd.rank()))
        epoch_start = time.time()
        total_train_loss = 0.0
        t_cnt = 0
        for i, (x, y, bucket_id, len_x, len_y) in enumerate(dl.iterate_one_epoch()):
          # do training
          if i % FLAGS.summary_frequency == 0 and hvd.rank()==0: # print arg
            loss, _, samples, sm, lr = sess.run(fetches=fetches_s,
                                        feed_dict={
                                          model.x: x,
                                          model.y: y,
                                          model.x_length: len_x,
                                          model.y_length: len_y
                                        })
            if hvd.rank() == 0:
              sw.add_summary(sm, global_step=sess.run(global_step))
            utils.deco_print("In epoch {}, step {} the loss is {}. Global step is {}".format(epoch, i, loss, sess.run(global_step)))
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

          # run evaluation if necessary
          if do_eval and i  % FLAGS.eval_frequency == 0:
            utils.deco_print("Evaluation on validation set")
            preds = []
            targets = []
            # iterate through evaluation data
            for j, (ex, ey, ebucket_id, elen_x, elen_y) in enumerate(eval_dl.iterate_one_epoch()):
              samples = sess.run(fetches=eval_fetches,
                                 feed_dict={
                                   e_model.x: ex,
                                   e_model.x_length: elen_x,
                                 })
              samples = samples[0].predicted_ids[:, :, 0] if eval_use_beam_search else samples[0].sample_id

              if eval_using_bleu:
                preds.extend([utils.transform_for_bleu(si,
                                                       vocab=eval_dl.target_idx2seq,
                                                       ignore_special=True,
                                                       delim=config["delimiter"], bpe_used=bpe_used) for sample in
                              [samples] for si in sample])
                targets.extend([[utils.transform_for_bleu(yi,
                                                          vocab=eval_dl.target_idx2seq,
                                                          ignore_special=True,
                                                          delim=config["delimiter"], bpe_used=bpe_used)] for yii in [ey]
                                for yi in yii])
            eval_dl.bucketize()
            if eval_using_bleu:
              eval_bleu = utils.calculate_bleu(preds, targets)
              bleu_value = summary_pb2.Summary.Value(tag="Eval_BLEU_Score", simple_value=eval_bleu)
              bleu_summary = summary_pb2.Summary(value=[bleu_value])
              sw.add_summary(summary=bleu_summary, global_step=sess.run(global_step))
              sw.flush()

            # if i > 0:
            #  utils.deco_print("Saving EVAL checkpoint")
            #  eval_saver.save(sess, save_path=os.path.join(FLAGS.logdir, "model-eval"), global_step=global_step)

        # epoch finished
        epoch_end = time.time()
        utils.deco_print('Epoch {} training loss: {}'.format(epoch, total_train_loss / t_cnt))
        if hvd.rank() == 0:
          value = summary_pb2.Summary.Value(tag="TrainEpochLoss", simple_value= total_train_loss / t_cnt)
          summary = summary_pb2.Summary(value=[value])
          sw.add_summary(summary=summary, global_step=epoch)
          sw.flush()
        utils.deco_print("Did epoch {} in {} seconds".format(epoch, epoch_end - epoch_start))
        dl.bucketize()
      # end of epoch loop

def main(_):
  with open(FLAGS.config_file) as data_file:
    in_config = json.load(data_file)
  if FLAGS.mode == "train":
    utils.deco_print("Running in training mode")
    train_config = utils.configure_params(in_config, "train")
    if 'source_file_eval' in in_config and 'target_file_eval' in in_config:
      eval_config = utils.configure_params(in_config, "eval")
      train(train_config, eval_config)
    else:
      train(train_config, None)
  else:
    raise ValueError("Unknown mode in config file. For inference, do not use Horovod")

if __name__ == "__main__":
  tf.app.run()
