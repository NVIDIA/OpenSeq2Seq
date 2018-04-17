# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from open_seq2seq.parts.common import get_pad_masking_bias, inf
from open_seq2seq.data.text2text import SpecialTextTokens


class Get_pad_masking_biasTest(tf.test.TestCase):
  def setUp(self):
    print("Setting Up  get_pad_masking_bias Test")

  def tearDown(self):
    print("Tear down  get_pad_masking_bias Test")

  def test_pad_masking_bias(self):
    batch_size = 1
    Q_len = 4
    K_len = 3
    heads = 1
    Q = tf.placeholder(dtype=tf.float32, shape=[batch_size, Q_len])
    K = tf.placeholder(dtype=tf.float32, shape=[batch_size, K_len])
    mask = get_pad_masking_bias(x=Q, y=K, PAD_ID=SpecialTextTokens.PAD_ID.value,
                                heads=heads)
    eQ = np.array([[0.3, 4.4, 4.5, SpecialTextTokens.PAD_ID.value]])
    eK = np.array([[0.3, 1.3, SpecialTextTokens.PAD_ID.value]])
    with self.test_session(use_gpu=True) as sess:
      e_mask = sess.run(mask, feed_dict={Q: eQ, K: eK})
      print(e_mask)
    self.assertAllEqual(e_mask, inf*np.array([[[[0., 0., 1.],
                                            [0., 0., 1.],
                                            [0., 0., 1.],
                                            [1., 1., 1.]]]]))