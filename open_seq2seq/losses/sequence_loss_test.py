# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

from open_seq2seq.losses.sequence_loss import CrossEntropyWithSmoothing, \
  BasicSequenceLoss


class CrossEntropyWithSmoothingEqualsBasicSequenceLossTest(tf.test.TestCase):
  def setUp(self):
    print("Setting Up  CrossEntropyWithSmoothingEqualsBasicSequenceLoss Test")

  def tearDown(self):
    print("Tear down  CrossEntropyWithSmoothingEqualsBasicSequenceLoss Test")

  def test_compute_loss(self):
    seq_length = 13
    tgt_vocab_size = 12

    for offset in [0, 3, 4]:
      for batch_size in [1, 4, 8]:
        for _o in [True, False]:
          for _m in [True, False]:
            loss_params = {
                "do_mask": _m,
                "tgt_vocab_size": tgt_vocab_size,
                "batch_size": batch_size,
                "offset_target_by_one": _o,
            }

            targets = tf.placeholder(dtype=tf.int32, shape=[batch_size,
                                                            seq_length])
            logits = tf.placeholder(dtype=tf.float32, shape=[batch_size,
                                                             seq_length,
                                                             tgt_vocab_size])
            tgt_lengths = tf.placeholder(dtype=tf.int32, shape=[batch_size])
            xentropy = CrossEntropyWithSmoothing(params=loss_params, model=None)
            sparse_xentropy = BasicSequenceLoss(params=loss_params, model=None)
            loss_input_dict = {
                "decoder_output": {"logits": logits},
                "target_tensors": [targets, tgt_lengths],
            }
            l1 = sparse_xentropy.compute_loss(input_dict=loss_input_dict)
            l2 = xentropy.compute_loss(input_dict=loss_input_dict)
            with self.test_session(use_gpu=True) as sess:
              tgts = np.random.randint(tgt_vocab_size,
                                       size=(batch_size, seq_length))
              lgts = np.random.random(size=[batch_size,
                                            seq_length, tgt_vocab_size])
              feed_dict = {
                  targets: tgts,
                  logits: lgts,
                  tgt_lengths: np.array([seq_length-offset]*batch_size)
              }
              loss1 = sess.run(l1, feed_dict=feed_dict)
              loss2 = sess.run(l2, feed_dict=feed_dict)
              self.assertEqual(loss1, loss2)
              print("Loss: {}".format(loss1))


if __name__ == '__main__':
  tf.test.main()
