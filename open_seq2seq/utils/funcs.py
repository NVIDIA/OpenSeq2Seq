# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import time

from .hooks import PrintSamplesHook, RunEvaluationHook, PrintLossAndTimeHook
from open_seq2seq.utils.utils import deco_print
from tensorflow.python import debug as tf_debug

def train(config,
          train_model,
          eval_model=None,
          hvd=None):
  """
  Training Loop function
  :param config:
  :param train_model:
  :param eval_model:
  :param hvd:
  :return:
  """
  if eval_model is not None and 'eval_frequency' not in config:
    raise ValueError("eval_frequency parameter has to be specified "
                     "if eval_model is provided")
  if hvd:
    master_worker = hvd.rank() == 0
  else:
    master_worker = True

  # initializing session parameters
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  if hvd is not None:
    sess_config.gpu_options.visible_device_list = str(hvd.local_rank())

  # defining necessary hooks
  hooks = [tf.train.StopAtStepHook(last_step=train_model.last_step)]
  if hvd is not None:
    hooks.append(hvd.BroadcastGlobalVariablesHook(0))

  if master_worker:
    checkpoint_dir = train_model.params['logdir']
  else:
    checkpoint_dir = None

  if master_worker:
    if config['checkpoint_frequency'] is not None:
      # noinspection PyTypeChecker
      hooks.append(tf.train.CheckpointSaverHook(
        checkpoint_dir, save_steps=config['checkpoint_frequency'])
      )
    if config['print_loss_frequency'] is not None:
      # noinspection PyTypeChecker
      hooks.append(PrintLossAndTimeHook(
        frequency=config['print_loss_frequency'],
        model=train_model,
      ))
    if config['print_samples_frequency'] is not None:
      # noinspection PyTypeChecker
      hooks.append(PrintSamplesHook(
        frequency=config['print_samples_frequency'],
        model=train_model,
      ))

  if eval_model is not None:
    # noinspection PyTypeChecker
    hooks.append(
      RunEvaluationHook(
        frequency=config['eval_frequency'],
        model=eval_model,
        last_step=train_model.last_step,
      ),
    )
  total_time = 0.0
  bench_start = config.get('bench_start', 10)

  #hooks.append(tf_debug.TensorBoardDebugHook("Trantor:6067"))
  # starting training
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=checkpoint_dir,
      save_summaries_steps=config['summary_frequency'],
      config=sess_config,
      save_checkpoint_secs=None,
      log_step_count_steps=config['summary_frequency'],
      stop_grace_period_secs=300,
      hooks=hooks) as sess:
    for step, feed_dict in enumerate(train_model.data_layer.iterate_forever()):
      if sess.should_stop():
        break
      tm = time.time()
      sess.run(fetches=train_model.train_op, feed_dict=feed_dict)
      if step >= bench_start:
        total_time += time.time() - tm

  if hvd is not None:
    deco_print("Finished training on rank {}".format(hvd.rank()))
  else:
    deco_print("Finished training")

  if step > bench_start:
    deco_print(
      "Avg time per step: {:.3}s".format(
        1.0 * total_time / (step - bench_start))
    )
  else:
    deco_print("Not enough steps for benchmarking")


def get_batches_for_epoch(model, checkpoint, config):
  total_time = 0.0
  bench_start = config.get('bench_start', 10)

  saver = tf.train.Saver()
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  with tf.Session(config=sess_config) as sess:
    saver.restore(sess, checkpoint)
    inputs_per_batch, outputs_per_batch = [], []
    fetches = [model.data_layer.get_input_tensors(), model.get_output_tensors()]
    total_batches = model.data_layer.get_size_in_batches()
    for step, feed_dict in enumerate(model.data_layer.iterate_one_epoch()):
      tm = time.time()
      inputs, outputs = sess.run(fetches, feed_dict)
      if step >= bench_start:
        total_time += time.time() - tm
      inputs_per_batch.append(inputs)
      outputs_per_batch.append(outputs)

      ending = '\r' if step < total_batches - 1 else '\n'
      deco_print("Processed {}/{} batches".format(step + 1, total_batches),
                 end=ending)
  if step > bench_start:
    deco_print(
      "Avg time per step: {:.3}s".format(
        1.0 * total_time / (step - bench_start))
    )
  else:
    deco_print("Not enough steps for benchmarking")
  return inputs_per_batch, outputs_per_batch


def infer(config, model, checkpoint, output_file):
  inputs_per_batch, outputs_per_batch = get_batches_for_epoch(model,
                                                              checkpoint,
                                                              config)
  model.infer(inputs_per_batch, outputs_per_batch, output_file)
  deco_print("Finished inference")


def evaluate(config, model, checkpoint):
  # TODO: last batch might be cut!
  inputs_per_batch, outputs_per_batch = get_batches_for_epoch(model,
                                                              checkpoint,
                                                              config)
  eval_dict = model.maybe_evaluate(inputs_per_batch, outputs_per_batch)
  deco_print("Finished evaluation")
  return eval_dict
