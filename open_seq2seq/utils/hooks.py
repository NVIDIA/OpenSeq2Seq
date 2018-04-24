# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import time
import os

from open_seq2seq.utils.utils import deco_print


def log_summaries_from_dict(dict_to_log, output_dir, step):
  # this returns the same writer as was created by
  # the first call to this function
  sm_writer = tf.summary.FileWriterCache.get(output_dir)
  for tag, value in dict_to_log.items():
    sm_writer.add_summary(
      tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)]),
      global_step=step,
    )
    sm_writer.flush()


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
    self._eval_saver = tf.train.Saver()
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

    deco_print("Running evaluation on a validation set:")

    inputs_per_batch, outputs_per_batch = [], []
    total_loss = 0.0

    for cnt, feed_dict in enumerate(self._model.data_layer.iterate_one_epoch()):
      loss, inputs, outputs = run_context.session.run(
        self._fetches, feed_dict,
      )
      inputs_per_batch.append(inputs)
      outputs_per_batch.append(outputs)
      total_loss += loss

    total_loss /= (cnt + 1)
    deco_print("Validation loss: {:.4f}".format(total_loss), offset=4)
    dict_to_log = self._model.maybe_evaluate(
      inputs_per_batch,
      outputs_per_batch,
    )
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
