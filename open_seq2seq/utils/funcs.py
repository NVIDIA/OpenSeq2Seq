# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
import time

from .hooks import PrintSamplesHook, RunEvaluationHook, PrintLossAndTimeHook
from open_seq2seq.utils.utils import deco_print, get_results_for_epoch
from tensorflow.python import debug as tf_debug


def train(train_model, eval_model=None, debug_port=None):
  if eval_model is not None and 'eval_steps' not in eval_model.params:
    raise ValueError("eval_steps parameter has to be specified "
                     "if eval_model is provided")
  hvd = train_model.hvd
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

  if eval_model is not None:
    # noinspection PyTypeChecker
    hooks.append(
      RunEvaluationHook(
        every_steps=eval_model.params['eval_steps'],
        model=eval_model,
        last_step=train_model.last_step,
      ),
    )

  if master_worker:
    if train_model.params['save_checkpoint_steps'] is not None:
      # noinspection PyTypeChecker
      saver = tf.train.Saver(save_relative_paths=True)
      hooks.append(tf.train.CheckpointSaverHook(
        checkpoint_dir,
        saver=saver,
        save_steps=train_model.params['save_checkpoint_steps']),
      )
    if train_model.params['print_loss_steps'] is not None:
      # noinspection PyTypeChecker
      hooks.append(PrintLossAndTimeHook(
        every_steps=train_model.params['print_loss_steps'],
        model=train_model,
      ))
    if train_model.params['print_samples_steps'] is not None:
      # noinspection PyTypeChecker
      hooks.append(PrintSamplesHook(
        every_steps=train_model.params['print_samples_steps'],
        model=train_model,
      ))

  total_time = 0.0
  bench_start = train_model.params.get('bench_start', 10)

  if debug_port:
    hooks.append(
      tf_debug.TensorBoardDebugHook("localhost:{}".format(debug_port))
    )

  # starting training
  with tf.train.MonitoredTrainingSession(
    checkpoint_dir=checkpoint_dir,
    save_summaries_steps=train_model.params['save_summaries_steps'],
    config=sess_config,
    save_checkpoint_secs=None,
    log_step_count_steps=train_model.params['save_summaries_steps'],
    stop_grace_period_secs=300,
    hooks=hooks,
  ) as sess:
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


def restore_and_get_results(model, checkpoint, mode):
  saver = tf.train.Saver()
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  if model.hvd:
    sess_config.gpu_options.visible_device_list = str(model.hvd.local_rank())
  with tf.Session(config=sess_config) as sess:
    saver.restore(sess, checkpoint)
    results_per_batch = get_results_for_epoch(
      model, sess, mode=mode, compute_loss=False, verbose=True,
    )
  return results_per_batch


def infer(model, checkpoint, output_file):
  results_per_batch = restore_and_get_results(model, checkpoint, mode="infer")
  if not model.on_horovod or model.hvd.rank() == 0:
    model.finalize_inference(results_per_batch, output_file)
    deco_print("Finished inference")


def evaluate(model, checkpoint):
  results_per_batch = restore_and_get_results(model, checkpoint, mode="eval")
  if not model.on_horovod or model.hvd.rank() == 0:
    eval_dict = model.finalize_evaluation(results_per_batch)
    deco_print("Finished evaluation")
    return eval_dict
  else:
    return None
