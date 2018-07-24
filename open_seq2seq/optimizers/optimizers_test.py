# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import numpy.testing as npt
import tensorflow as tf
from six.moves import range

from open_seq2seq.optimizers import optimize_loss
from .lr_policies import fixed_lr


class IterSizeTests(tf.test.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_updates(self):
    try:
      import horovod.tensorflow as hvd
      hvd.init()
    except ImportError:
      print("Horovod not installed skipping test_updates")
      return

    dtype = tf.float32
    with tf.Graph().as_default() as g:
      n_samples = 10
      n_hid = 10
      var_dtype = tf.float32 if dtype == tf.float32 else tf.float16

      np.random.seed(0)
      X = np.random.rand(n_samples, n_hid)
      y = np.random.rand(n_samples, 1)
      w = np.linalg.solve(X.T.dot(X), X.T.dot(y))

      x_ph = tf.placeholder(var_dtype, [n_samples, n_hid])
      y_ph = tf.placeholder(var_dtype, [n_samples, 1])

      y_pred = tf.layers.dense(x_ph, 1, use_bias=False)
      loss = tf.losses.mean_squared_error(y_ph, y_pred)
      loss += tf.losses.get_regularization_loss()
      skip_update_ph = tf.placeholder(tf.bool)
      iter_size = 8
      train_op = optimize_loss(loss, "SGD", {},
                               lambda gs: fixed_lr(gs, 0.1), dtype=dtype,
                               iter_size=iter_size, on_horovod=True,
                               skip_update_ph=skip_update_ph)
      grad_accum = [var for var in tf.global_variables() if 'accum' in var.name][0]
      var = tf.trainable_variables()[0]
      with self.test_session(g, use_gpu=True) as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(3):
          g, v = sess.run([grad_accum, var])
          npt.assert_allclose(g, np.zeros(g.shape))

          true_g = 2 * (X.T.dot(X).dot(v) - X.T.dot(y)) / X.shape[0] / iter_size

          sess.run(train_op, {x_ph: X, y_ph: y, skip_update_ph: True})
          g_new, v_new = sess.run([grad_accum, var])
          npt.assert_allclose(g_new, true_g, atol=1e-7)
          npt.assert_allclose(v_new, v)

          sess.run(train_op, {x_ph: X, y_ph: y, skip_update_ph: True})
          g_new, v_new = sess.run([grad_accum, var])
          npt.assert_allclose(g_new, true_g * 2, atol=1e-7)
          npt.assert_allclose(v_new, v)

          sess.run(train_op, {x_ph: X, y_ph: y, skip_update_ph: True})
          g_new, v_new = sess.run([grad_accum, var])
          npt.assert_allclose(g_new, true_g * 3, atol=1e-7)
          npt.assert_allclose(v_new, v)

          sess.run(train_op, {x_ph: X, y_ph: y, skip_update_ph: False})
          g_new, v_new = sess.run([grad_accum, var])
          npt.assert_allclose(g_new, np.zeros(g.shape))
          npt.assert_allclose(v_new, v - 0.1 * true_g * 4, atol=1e-7)


if __name__ == '__main__':
  tf.test.main()
