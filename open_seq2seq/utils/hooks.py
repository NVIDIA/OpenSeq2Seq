# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
import time
import os


from open_seq2seq.utils.utils import deco_print, log_summaries_from_dict, \
                                     get_results_for_epoch


class PrintSamplesHook(tf.train.SessionRunHook):
  """Session hook that prints training samples and prediction from time to time
  """
  def __init__(self, every_steps, model):
    super(PrintSamplesHook, self).__init__()
    self._timer = tf.train.SecondOrStepTimer(every_steps=every_steps)
    self._iter_count = 0
    self._global_step = None
    self._model = model
    self._fetches = [
      model.data_layer.get_input_tensors(),
      model.get_output_tensors(),
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
    if not self._model.on_horovod:
      # clipping to only first GPU
      input_values = input_values[0]
      output_values = output_values[0]
    dict_to_log = self._model.maybe_print_logs(input_values, output_values)
    # optionally logging to tensorboard any values
    # returned from maybe_print_logs
    if dict_to_log:
      log_summaries_from_dict(
        dict_to_log,
        self._model.params['logdir'],
        step,
      )


class PrintLossAndTimeHook(tf.train.SessionRunHook):
  """Session hook that prints training samples and prediction from time to time
  """
  def __init__(self, every_steps, model):
    super(PrintLossAndTimeHook, self).__init__()
    self._timer = tf.train.SecondOrStepTimer(every_steps=every_steps)
    self._every_steps = every_steps
    self._iter_count = 0
    self._global_step = None
    self._model = model
    self._fetches = [model.loss]
    self._last_time = time.time()

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
    deco_print("loss = {:.4f}".format(loss), start="", end=", ")

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
  def __init__(self, every_steps, model, last_step=-1):
    super(RunEvaluationHook, self).__init__()
    self._timer = tf.train.SecondOrStepTimer(every_steps=every_steps)
    self._iter_count = 0
    self._global_step = None
    self._model = model
    self._fetches = [
      model.loss,
      model.data_layer.get_input_tensors(),
      model.get_output_tensors(),
    ]
    self._triggered = False
    self._last_step = last_step
    self._eval_saver = tf.train.Saver(save_relative_paths=True)
    self._best_eval_loss = 1e9

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
      deco_print("Validation loss: {:.4f}".format(total_loss), offset=4)

      dict_to_log = self._model.finalize_evaluation(results_per_batch)
      dict_to_log['eval_loss'] = total_loss

      # saving the best validation model
      if total_loss < self._best_eval_loss:
        self._best_eval_loss = total_loss
        self._eval_saver.save(
          run_context.session,
          os.path.join(self._model.params['logdir'], 'best_models',
                       'val_loss={:.4f}-step'.format(total_loss)),
          global_step=step + 1,
        )

      # optionally logging to tensorboard any values
      # returned from maybe_print_logs
      if dict_to_log:
        log_summaries_from_dict(
          dict_to_log,
          self._model.params['logdir'],
          step,
        )
