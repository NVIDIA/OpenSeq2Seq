# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
import numpy as np
from open_seq2seq.parts.attention import get_future_masking_bias

class Get_future_masking_biasTest(tf.test.TestCase):
  def setUp(self):
    print("Setting Up  Get_future_masking_bias Test")

  def tearDown(self):
    print("Tear down  Get_future_masking_bias Test")

  def test_future_masking_bias(self):
    batch, num_heads, Q_length, K_length, dk = 1, 1, 6, 6, 22
    dtype = tf.float32

    Q = tf.placeholder(dtype=dtype, shape=[batch, num_heads, Q_length, dk])
    K = tf.placeholder(dtype=dtype, shape=[batch, num_heads, K_length, dk])
    bias = tf.nn.softmax(get_future_masking_bias(Q, K))

    feed_dict = {Q: np.random.random(size=(batch, num_heads, Q_length, dk)),
                 K: np.random.random(size=(batch, num_heads, K_length, dk))}

    with self.test_session(use_gpu=True) as sess:
      eb = sess.run(bias, feed_dict=feed_dict)
    print(eb[0, 0])