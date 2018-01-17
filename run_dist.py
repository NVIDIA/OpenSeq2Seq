# Copyright (c) 2017 NVIDIA Corporation
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import tensorflow as tf
import math
from open_seq2seq.model import seq2seq_model
from open_seq2seq.model.model_utils import SaveAtEnd
from open_seq2seq.data import data_layer, utils
import horovod.tensorflow as hvd

tf.flags.DEFINE_string("config_file", "",
                       """Path to the file with configuration""")
tf.flags.DEFINE_string("logdir", "",
                       """Path to where save logs and checkpoints""")
tf.flags.DEFINE_integer("checkpoint_frequency", 7200,
                       """How often (in seconds) to save checkpoints""")
tf.flags.DEFINE_integer("summary_frequency", 50,
                       """summary step frequencey save rate""")
tf.flags.DEFINE_integer("max_steps", 300000,
                        """maximum training steps""")
tf.flags.DEFINE_string("mode", "train",
                       """Mode: train - for training mode, infer - for inference mode""")
tf.flags.DEFINE_boolean("split_data_per_rank", False,
                        "splits data between horovod ranks. may use less ram")
tf.flags.DEFINE_float("lr", None,
                      "If not None, this will overwrite learning rate in config")

FLAGS = tf.flags.FLAGS

def train(config):
  """
  Implements training mode
  :param config: python dictionary describing model and data layer
  :param eval_config: (default) None python dictionary describing model and data layer used for evaluation
  :return: nothing
  """
  hvd.init()
  utils.deco_print("Executing training mode")
  utils.deco_print("Creating data layer")
  if FLAGS.split_data_per_rank:
    dl = data_layer.ParallelDataInRamInputLayer(params = config,
                                                num_workers = hvd.size(),
                                                worker_id = hvd.rank())
  else:
    dl = data_layer.ParallelDataInRamInputLayer(params=config)
  if 'pad_vocabs_to_eight' in config and config['pad_vocabs_to_eight']:
    config['src_vocab_size'] = int(math.ceil(len(dl.source_seq2idx) / 8) * 8)
    config['tgt_vocab_size'] = int(math.ceil(len(dl.target_seq2idx) / 8) * 8)
  else:
    config['src_vocab_size'] = len(dl.source_seq2idx)
    config['tgt_vocab_size'] = len(dl.target_seq2idx)
  utils.deco_print("Data layer created")

  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Create train model
    model = seq2seq_model.BasicSeq2SeqWithAttention(model_params = config,
                                                    global_step = global_step,
                                                    mode = "train",
                                                    gpu_ids = "horovod")
    fetches = [model.loss, model.train_op, model.lr]

    if hvd.rank()==0:
      tf.summary.scalar(name="loss", tensor=model.loss)
      summary_op = tf.summary.merge_all()
      fetches_s = [model.loss, model.train_op, model.final_outputs, summary_op, model.lr]
      #sw = tf.summary.FileWriter(logdir=FLAGS.logdir, flush_secs=60)
    # done constructing graph at this point

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess_config.gpu_options.allow_growth = True

    hooks = [hvd.BroadcastGlobalVariablesHook(hvd.size()-1),
             tf.train.StepCounterHook(every_n_steps=FLAGS.summary_frequency),
             tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
             SaveAtEnd(logdir=FLAGS.logdir, global_step=global_step)]
    checkpoint_dir = FLAGS.logdir if hvd.rank() == 0 else None

    with tf.train.MonitoredTrainingSession(checkpoint_dir = checkpoint_dir,
                                           save_summaries_steps = FLAGS.summary_frequency,
                                           config = sess_config,
                                           save_checkpoint_secs = FLAGS.checkpoint_frequency,
                                           log_step_count_steps = FLAGS.summary_frequency,
                                           stop_grace_period_secs = 300,
                                           hooks = hooks) as sess:
      #begin training
      for i, (x, y, bucket_id, len_x, len_y) in enumerate(dl.iterate_forever()):
        if not sess.should_stop():
          # do training
          if i % FLAGS.summary_frequency == 0 and hvd.rank() == 0: # print arg
            loss, _, samples, sm, lr = sess.run(fetches=fetches_s,
                                        feed_dict={
                                          model.x: x,
                                          model.y: y,
                                          model.x_length: len_x,
                                          model.y_length: len_y
                                        })
            #sw.add_summary(sm, global_step=sess.run(global_step))
            utils.deco_print("Step: " + str(i))
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
          # training step done
        else:
          utils.deco_print("Finished training on rank {}".format(hvd.rank()))
          break

def main(_):
  with open(FLAGS.config_file) as data_file:
    in_config = json.load(data_file)
    if FLAGS.lr is not None:
      in_config["learning_rate"] = FLAGS.lr
      utils.deco_print("using LR from command line: {}".format(FLAGS.lr))
    if 'num_gpus' in in_config:
      utils.deco_print("num_gpus parameters is ignored when using Horovod")
    if 'num_epochs' in in_config:
      utils.deco_print("num_epochs parameters is ignored when using Horovod. Instead use --max_steps")

  if FLAGS.mode == "train":
    utils.deco_print("Running in training mode")
    train_config = utils.configure_params(in_config, "train")
    if 'source_file_eval' in in_config and 'target_file_eval' in in_config:
      utils.deco_print("Eval is not supported when using Horovod")
    train(train_config)
  else:
    raise ValueError("Unknown mode in config file. For inference, do not use Horovod")

if __name__ == "__main__":
  tf.app.run()
