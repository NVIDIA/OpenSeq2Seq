# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import math
import os
import time

import tensorflow as tf

from open_seq2seq.utils.utils import deco_print, log_summaries_from_dict, \
                                     get_results_for_epoch


class BroadcastGlobalVariablesHook(tf.train.SessionRunHook):
  """
  SessionRunHook that will broadcast all global variables from root rank
  to all other processes during initialization.
  This is necessary to ensure consistent initialization of all workers when
  training is started with random weights or restored from a checkpoint.
  """

  def __init__(self, root_rank, device=''):
    """Construct a new BroadcastGlobalVariablesHook that will broadcast all
    global variables from root rank to all other processes during initialization.
    Args:
      root_rank:
        Rank that will send data, other ranks will receive data.
      device:
        Device to be used for broadcasting. Uses GPU by default
        if Horovod was build with HOROVOD_GPU_BROADCAST.
    """
    super(BroadcastGlobalVariablesHook, self).__init__()
    self.root_rank = root_rank
    self.bcast_op = None
    self.device = device

  def begin(self):
    def broadcast_global_variables(root_rank):
      from horovod.tensorflow.mpi_ops import broadcast
      ops = []
      for var in tf.global_variables():
        if var.dtype.base_dtype == tf.float16:
          ops.append(tf.assign(var, tf.cast(broadcast(tf.cast(var, tf.float32),
                                                      root_rank), tf.float16)))
        else:
          ops.append(tf.assign(var, broadcast(var, root_rank)))
      return tf.group(*ops)

    if not self.bcast_op or self.bcast_op.graph != tf.get_default_graph():
      with tf.device(self.device):
        self.bcast_op = broadcast_global_variables(self.root_rank)

  def after_create_session(self, session, coord):
    session.run(self.bcast_op)


class PrintSamplesHook(tf.train.SessionRunHook):
  """Session hook that prints training samples and prediction from time to time
  """
  def __init__(self, every_steps, model):
    super(PrintSamplesHook, self).__init__()
    self._timer = tf.train.SecondOrStepTimer(every_steps=every_steps)
    self._iter_count = 0
    self._global_step = None
    self._model = model
    # using only first GPU
    output_tensors = model.get_output_tensors(0)
    self._fetches = [
        model.get_data_layer(0).input_tensors,
        output_tensors,
    ]

  def begin(self):
    self._iter_count = 0
    self._global_step = tf.train.get_global_step()

  def before_run(self, run_context):
    if self._timer.should_trigger_for_step(self._iter_count):
      return tf.train.SessionRunArgs([self._fetches, self._global_step])
    return tf.train.SessionRunArgs([[], self._global_step])

  def after_run(self, run_context, run_values):
    results, step = run_values.results
    self._iter_count = step

    if not results:
      return
    self._timer.update_last_triggered_step(self._iter_count - 1)

    input_values, output_values = results
    dict_to_log = self._model.maybe_print_logs(input_values, output_values, step)
    # optionally logging to tensorboard any values
    # returned from maybe_print_logs
    if self._model.params['save_summaries_steps'] and dict_to_log:
      log_summaries_from_dict(
          dict_to_log,
          self._model.params['logdir'],
          step,
      )


class PrintLossAndTimeHook(tf.train.SessionRunHook):
  """Session hook that prints training samples and prediction from time to time
  """
  def __init__(self, every_steps, model, print_ppl=False):
    super(PrintLossAndTimeHook, self).__init__()
    self._timer = tf.train.SecondOrStepTimer(every_steps=every_steps)
    self._every_steps = every_steps
    self._iter_count = 0
    self._global_step = None
    self._model = model
    self._fetches = [model.loss]
    self._last_time = time.time()
    self._print_ppl = print_ppl

  def begin(self):
    self._iter_count = 0
    self._global_step = tf.train.get_global_step()

  def before_run(self, run_context):
    if self._timer.should_trigger_for_step(self._iter_count):
      return tf.train.SessionRunArgs([self._fetches, self._global_step])
    return tf.train.SessionRunArgs([[], self._global_step])

  def after_run(self, run_context, run_values):
    results, step = run_values.results
    self._iter_count = step

    if not results:
      return
    self._timer.update_last_triggered_step(self._iter_count - 1)

    if self._model.steps_in_epoch is None:
      deco_print("Global step {}:".format(step), end=" ")
    else:
      deco_print(
          "Epoch {}, global step {}:".format(
              step // self._model.steps_in_epoch, step),
          end=" ",
      )

    loss = results[0]
    if not self._model.on_horovod or self._model.hvd.rank() == 0:
      if self._print_ppl:
        deco_print("loss: {:.4f} | ppl = {:.4f} | bpc = {:.4f}"
                   .format(loss, math.exp(loss),
                           loss/math.log(2)),
                   start="", end=", ")
      else:
        deco_print("loss: {:.4f} ".format(loss), start="", end=", ")

    tm = (time.time() - self._last_time) / self._every_steps
    m, s = divmod(tm, 60)
    h, m = divmod(m, 60)

    deco_print(
        "time per step = {}:{:02}:{:.3f}".format(int(h), int(m), s),
        start="",
    )
    self._last_time = time.time()


class RunEvaluationHook(tf.train.SessionRunHook):
  """Session hook that runs evaluation on a validation set
  """
  def __init__(self, every_steps, model, last_step=-1, print_ppl=False):
    super(RunEvaluationHook, self).__init__()
    self._timer = tf.train.SecondOrStepTimer(every_steps=every_steps)
    self._iter_count = 0
    self._global_step = None
    self._model = model
    self._triggered = False
    self._last_step = last_step
    self._eval_saver = tf.train.Saver(save_relative_paths=True)
    self._best_eval_loss = 1e9
    self._print_ppl = print_ppl

  def begin(self):
    self._iter_count = 0
    self._global_step = tf.train.get_global_step()

  def before_run(self, run_context):
    self._triggered = self._timer.should_trigger_for_step(self._iter_count)
    return tf.train.SessionRunArgs([[], self._global_step])

  def after_run(self, run_context, run_values):
    results, step = run_values.results
    self._iter_count = step

    if not self._triggered and step != self._last_step - 1:
      return
    self._timer.update_last_triggered_step(self._iter_count - 1)

    if not self._model.on_horovod or self._model.hvd.rank() == 0:
      deco_print("Running evaluation on a validation set:")

    results_per_batch, total_loss = get_results_for_epoch(
        self._model, run_context.session, mode="eval", compute_loss=True,
    )

    if not self._model.on_horovod or self._model.hvd.rank() == 0:
      if self._print_ppl:
        deco_print("Validation loss: {:.4f} | ppl = {:.4f} | bpc = {:.4f}"
                   .format(total_loss, math.exp(total_loss),
                           total_loss/math.log(2)), offset=4)
      else:
        deco_print(
          "Validation loss: {:.4f} ".format(total_loss),
          offset=4)


      dict_to_log = self._model.finalize_evaluation(results_per_batch, step)
      dict_to_log['eval_loss'] = total_loss

      # saving the best validation model
      if self._model.params['save_checkpoint_steps'] and \
         total_loss < self._best_eval_loss:
        self._best_eval_loss = total_loss
        self._eval_saver.save(
            run_context.session,
            os.path.join(self._model.params['logdir'], 'best_models',
                         'val_loss={:.4f}-step'.format(total_loss)),
            global_step=step + 1,
        )

      # optionally logging to tensorboard any values
      # returned from maybe_print_logs
      if self._model.params['save_summaries_steps']:
        log_summaries_from_dict(
            dict_to_log,
            self._model.params['logdir'],
            step,
        )
