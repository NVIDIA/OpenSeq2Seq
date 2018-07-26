# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import subprocess
import time

import numpy as np
import six
from six import string_types
from six.moves import range
import tensorflow as tf
# pylint: disable=no-name-in-module
from tensorflow.python.client import device_lib


def get_available_gpus():
  # WARNING: this method will take all the memory on all devices!
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']


def clip_sparse(value, size):
  dense_shape_clipped = value.dense_shape
  dense_shape_clipped[0] = size
  indices_clipped = []
  values_clipped = []
  for idx_tuple, val in zip(value.indices, value.values):
    if idx_tuple[0] < size:
      indices_clipped.append(idx_tuple)
      values_clipped.append(val)
  return tf.SparseTensorValue(np.array(indices_clipped),
                              np.array(values_clipped),
                              dense_shape_clipped)


def collect_if_horovod(value, hvd, mode='sum'):
  """Collects values from all workers if run on Horovod.
  Note, that on all workers except first this function will return None.

  Args:
    value: value to collect.
    hvd: horovod.tensorflow module or None
    mode: could be "sum", "mean" or "gather", indicating reduce_sum or gather.
        For "sum" and "mean" value has to be numerical, for "gather", value has
        to be iterable.

  Returns:
    collected results if run on Horovod or value otherwise.
  """
  if hvd is None:
    return value

  import mpi4py.rc
  mpi4py.rc.initialize = False
  from mpi4py import MPI

  values = MPI.COMM_WORLD.gather(value)
  # synchronize all workers
  MPI.COMM_WORLD.Barrier()

  if MPI.COMM_WORLD.Get_rank() != 0:
    return None

  if mode == 'sum':
    return np.sum(values)
  elif mode == 'mean':
    return np.mean(values)
  elif mode == 'gather':
    return [item for sl in values for item in sl]
  else:
    raise ValueError("Incorrect mode: {}".format(mode))


def clip_last_batch(last_batch, true_size):
  last_batch_clipped = []
  for val in last_batch:
    if isinstance(val, tf.SparseTensorValue):
      last_batch_clipped.append(clip_sparse(val, true_size))
    else:
      last_batch_clipped.append(val[:true_size])
  return last_batch_clipped


def iterate_data(model, sess, compute_loss, mode, verbose, input=None):
  total_time = 0.0
  bench_start = model.params.get('bench_start', 10)
  results_per_batch = []

  size_defined = model.get_data_layer().get_size_in_samples() is not None
  if size_defined:
    dl_sizes = []

  if compute_loss:
    total_loss = 0.0

  total_samples = []
  fetches = []

  # on horovod num_gpus is 1
  for worker_id in range(model.num_gpus):
    cur_fetches = [
        model.get_data_layer(worker_id).input_tensors,
        model.get_output_tensors(worker_id),
    ]
    if compute_loss:
      cur_fetches.append(model.eval_losses[worker_id])
    if size_defined:
      dl_sizes.append(model.get_data_layer(worker_id).get_size_in_samples())
    try:
      total_objects = 0.0
      cur_fetches.append(model.get_num_objects_per_step(worker_id))
    except NotImplementedError:
      total_objects = None
      deco_print("WARNING: Can't compute number of objects per step, since "
                 "train model does not define get_num_objects_per_step method.")
    fetches.append(cur_fetches)
    total_samples.append(0.0)

  if mode == "interactive_infer":
    sess.run(
        [model.get_data_layer().iterator.initializer],
        feed_dict={model.get_data_layer().input:input}
    )
    mode = "infer"
  else:
    sess.run([model.get_data_layer(i).iterator.initializer
              for i in range(model.num_gpus)])

  step = 0
  processed_batches = 0
  if verbose:
    if model.on_horovod:
      ending = " on worker {}".format(model.hvd.rank())
    else:
      ending = ""

  while True:
    tm = time.time()
    fetches_vals = {}
    if size_defined:
      fetches_to_run = {}
      # removing finished data layers
      for worker_id in range(model.num_gpus):
        if total_samples[worker_id] < dl_sizes[worker_id]:
          fetches_to_run[worker_id] = fetches[worker_id]
      fetches_vals = sess.run(fetches_to_run)
    else:
      # if size is not defined we have to process fetches sequentially, so not
      # to lose data when exception is thrown on one data layer
      for worker_id, one_fetch in enumerate(fetches):
        try:
          fetches_vals[worker_id] = sess.run(one_fetch)
        except tf.errors.OutOfRangeError:
          continue

    if step >= bench_start:
      total_time += time.time() - tm

    # looping over num_gpus. In Horovod case this loop is "dummy",
    # since num_gpus = 1
    for worker_id, fetches_val in fetches_vals.items():
      if compute_loss:
        inputs, outputs, loss = fetches_val[:3]
      else:
        inputs, outputs = fetches_val[:2]

      if total_objects is not None:
        total_objects += np.sum(fetches_val[-1])

      # assuming any element of inputs["source_tensors"] .shape[0] is batch size
      batch_size = inputs["source_tensors"][0].shape[0]
      total_samples[worker_id] += batch_size

      if size_defined:
        # this data_layer is at the last batch with few more elements, cutting
        if total_samples[worker_id] > dl_sizes[worker_id]:
          last_batch_size = dl_sizes[worker_id] % batch_size
          for key, value in inputs.items():
            inputs[key] = model.clip_last_batch(value, last_batch_size)
          outputs = model.clip_last_batch(outputs, last_batch_size)

      processed_batches += 1

      if compute_loss:
        total_loss += loss * batch_size

      if mode == 'eval':
        results_per_batch.append(model.evaluate(inputs, outputs))
      elif mode == 'infer':
        results_per_batch.append(model.infer(inputs, outputs))
      else:
        raise ValueError("Unknown mode: {}".format(mode))

    if verbose:
      if size_defined:
        data_size = int(np.sum(np.ceil(np.array(dl_sizes) /
                                       model.params['batch_size_per_gpu'])))
        if step == 0 or len(fetches_vals) == 0 or \
           (data_size > 10 and processed_batches % (data_size // 10) == 0):
          deco_print("Processed {}/{} batches{}".format(
              processed_batches, data_size, ending
          ))
      else:
        deco_print("Processed {} batches{}".format(processed_batches, ending),
                   end='\r')

    if len(fetches_vals) == 0:
      break
    step += 1

  if verbose:
    if step > bench_start:
      deco_print(
          "Avg time per step{}: {:.3}s".format(
              ending, 1.0 * total_time / (step - bench_start)
          ),
      )
      if total_objects is not None:
        avg_objects = 1.0 * total_objects / total_time
        deco_print("Avg objects per second{}: {:.3f}".format(ending,
                                                             avg_objects))
    else:
      deco_print("Not enough steps for benchmarking{}".format(ending))

  if compute_loss:
    return results_per_batch, total_loss, np.sum(total_samples)
  else:
    return results_per_batch


def get_results_for_epoch(
    model,
    sess,
    compute_loss,
    mode,
    verbose=False,
    input=None
):
  if compute_loss:
    results_per_batch, total_loss, total_samples = iterate_data(
        model, sess, compute_loss, mode, verbose, input
    )
  else:
    results_per_batch = iterate_data(
        model, sess, compute_loss, mode, verbose, input
    )

  if compute_loss:
    total_samples = collect_if_horovod(total_samples, model.hvd, 'sum')
    total_loss = collect_if_horovod(total_loss, model.hvd, 'sum')
  results_per_batch = collect_if_horovod(results_per_batch, model.hvd, 'gather')

  if results_per_batch is None:
    # returning dummy tuple of correct shape if not in master worker
    if compute_loss:
      return None, None
    else:
      return None

  if compute_loss:
    return results_per_batch, total_loss / total_samples
  else:
    return results_per_batch


def log_summaries_from_dict(dict_to_log, output_dir, step):
  """
  A function that writes values from dict_to_log to a tensorboard
  log file inside output_dir.

  Args:
    dict_to_log (dict):
      A dictiontary containing the tags and scalar values to log.
      The dictionary values could also contain tf.Summary.Value objects
      to support logging of image and audio data. In this mode, the
      dictionary key is ignored, as tf.Summary.Value already contains a
      tag.
    output_dir (str): dir containing the tensorboard file
    step (int): current training step
  """
  sm_writer = tf.summary.FileWriterCache.get(output_dir)
  for tag, value in dict_to_log.items():
    if isinstance(value, tf.Summary.Value):
      sm_writer.add_summary(
          tf.Summary(value=[value]),
          global_step=step,
      )
    else:
      sm_writer.add_summary(
          tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)]),
          global_step=step,
      )
    sm_writer.flush()


def get_git_hash():
  try:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   stderr=subprocess.STDOUT).decode()
  except subprocess.CalledProcessError as e:
    return "{}\n".format(e.output.decode("utf-8"))


def get_git_diff():
  try:
    return subprocess.check_output(['git', 'diff'],
                                   stderr=subprocess.STDOUT).decode()
  except subprocess.CalledProcessError as e:
    return "{}\n".format(e.output.decode("utf-8"))


class Logger(object):
  def __init__(self, stream, log_file):
    self.stream = stream
    self.log = log_file

  def write(self, msg):
    self.stream.write(msg)
    self.log.write(msg)

  def flush(self):
    self.stream.flush()
    self.log.flush()


def flatten_dict(dct):
  flat_dict = {}
  for key, value in dct.items():
    if isinstance(value, (int, float, string_types, bool)):
      flat_dict.update({key: value})
    elif isinstance(value, dict):
      flat_dict.update(
          {key + '/' + k: v for k, v in flatten_dict(dct[key]).items()}
      )
  return flat_dict


def nest_dict(flat_dict):
  nst_dict = {}
  for key, value in flat_dict.items():
    nest_keys = key.split('/')
    cur_dict = nst_dict
    for i in range(len(nest_keys) - 1):
      if nest_keys[i] not in cur_dict:
        cur_dict[nest_keys[i]] = {}
      cur_dict = cur_dict[nest_keys[i]]
    cur_dict[nest_keys[-1]] = value
  return nst_dict


def nested_update(org_dict, upd_dict):
  for key, value in upd_dict.items():
    if isinstance(value, dict):
      if key in org_dict:
        if not isinstance(org_dict[key], dict):
          raise ValueError(
              "Mismatch between org_dict and upd_dict at node {}".format(key)
          )
        nested_update(org_dict[key], value)
      else:
        org_dict[key] = value
    else:
      org_dict[key] = value


def mask_nans(x):
  x_zeros = tf.zeros_like(x)
  x_mask = tf.is_finite(x)
  y = tf.where(x_mask, x, x_zeros)
  return y


def deco_print(line, offset=0, start="*** ", end='\n'):
  if six.PY2:
    print((start + " " * offset + line).encode('utf-8'), end=end)
  else:
    print(start + " " * offset + line, end=end)


def array_to_string(row, vocab, delim=' '):
  n = len(vocab)
  return delim.join(map(lambda x: vocab[x], [r for r in row if 0 <= r < n]))


def text_ids_to_string(row, vocab, S_ID, EOS_ID, PAD_ID,
                       ignore_special=False, delim=' '):
  """For _-to-text outputs this function takes a row with ids,
  target vocabulary and prints it as a human-readable string
  """
  n = len(vocab)
  if ignore_special:
    f_row = []
    for char_id in row:
      if char_id == EOS_ID:
        break
      if char_id != PAD_ID and char_id != S_ID:
        f_row += [char_id]
    return delim.join(map(lambda x: vocab[x], [r for r in f_row if 0 < r < n]))
  else:
    return delim.join(map(lambda x: vocab[x], [r for r in row if 0 < r < n]))


def check_params(config, required_dict, optional_dict):
  if required_dict is None or optional_dict is None:
    return

  for pm, vals in required_dict.items():
    if pm not in config:
      raise ValueError("{} parameter has to be specified".format(pm))
    else:
      if vals == str:
        vals = string_types
      if vals and isinstance(vals, list) and config[pm] not in vals:
        raise ValueError("{} has to be one of {}".format(pm, vals))
      if vals and not isinstance(vals, list) and not isinstance(config[pm], vals):
        raise ValueError("{} has to be of type {}".format(pm, vals))

  for pm, vals in optional_dict.items():
    if vals == str:
      vals = string_types
    if pm in config:
      if vals and isinstance(vals, list) and config[pm] not in vals:
        raise ValueError("{} has to be one of {}".format(pm, vals))
      if vals and not isinstance(vals, list) and not isinstance(config[pm], vals):
        raise ValueError("{} has to be of type {}".format(pm, vals))

  for pm in config:
    if pm not in required_dict and pm not in optional_dict:
      raise ValueError("Unknown parameter: {}".format(pm))


def cast_types(input_dict, dtype):
  cast_input_dict = {}
  for key, value in input_dict.items():
    if isinstance(value, tf.Tensor):
      if value.dtype == tf.float16 or value.dtype == tf.float32:
        if value.dtype.base_dtype != dtype.base_dtype:
          cast_input_dict[key] = tf.cast(value, dtype)
          continue
    if isinstance(value, dict):
      cast_input_dict[key] = cast_types(input_dict[key], dtype)
      continue
    if isinstance(value, list):
      cur_list = []
      for nest_value in value:
        if isinstance(nest_value, tf.Tensor):
          if nest_value.dtype == tf.float16 or nest_value.dtype == tf.float32:
            if nest_value.dtype.base_dtype != dtype.base_dtype:
              cur_list.append(tf.cast(nest_value, dtype))
              continue
        cur_list.append(nest_value)
      cast_input_dict[key] = cur_list
      continue
    cast_input_dict[key] = input_dict[key]
  return cast_input_dict
  