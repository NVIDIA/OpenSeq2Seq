
# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# THIS CODE WAS TAKEN from: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_layers.py
# The original license is in the top of this file

import tensorflow as tf
import math

def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def get_timing_signal(length,
                      min_timescale=1,
                      max_timescale=1e4,
                      num_timescales=16,
                      dtype=tf.float32):
  """Create Tensor of sinusoids of different frequencies.

  Args:
    length: Length of the Tensor to create, i.e. Number of steps.
    min_timescale: a float
    max_timescale: a float
    num_timescales: an int

  Returns:
    Tensor of shape (length, 2*num_timescales)
  """
  #positions = tf.to_float(tf.range(length))
  positions = tf.cast(tf.range(length), dtype=dtype)
  log_timescale_increment = (
      math.log(max_timescale / min_timescale) / (num_timescales - 1))
  inv_timescales = min_timescale * tf.exp(
      #tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
      tf.cast(tf.range(num_timescales), dtype=dtype) * -log_timescale_increment)
  scaled_time = tf.expand_dims(positions, 1) * tf.expand_dims(inv_timescales, 0)
  return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)


def add_timing_signal(x, min_timescale=1, max_timescale=1e4, num_timescales=2):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.

  This allows attention to learn to use absolute and relative positions.
  The timing signal should be added to some precursor of both the source
  and the target of the attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  experessed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the depth dimension, padded with zeros to be the same depth as the input,
  and added into input.

  Args:
    x: a Tensor with shape [?, length, depth]
    min_timescale: a float
    max_timescale: a float
    num_timescales: an int <= depth/2

  Returns:
    a Tensor the same shape as x.
  """
  length = shape_list(x)[1]
  depth = shape_list(x)[2]
  signal = get_timing_signal(length, min_timescale, max_timescale,
                             num_timescales, dtype=x.dtype)
  padded_signal = tf.pad(signal, [[0, 0], [0, depth - 2 * num_timescales]])
  return x + tf.reshape(padded_signal, [1, length, depth])