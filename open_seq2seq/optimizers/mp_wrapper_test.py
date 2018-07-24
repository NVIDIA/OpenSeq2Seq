# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import numpy.testing as npt
import tensorflow as tf
from six.moves import range

from open_seq2seq.optimizers import optimize_loss
from open_seq2seq.optimizers.mp_wrapper import mp_regularizer_wrapper, \
                                               MixedPrecisionOptimizerWrapper
from .lr_policies import fixed_lr


class MixedPrecisionOptimizerTests(tf.test.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_regularization_normal(self):
    n_samples = 3
    n_hid = 2
    scale_init = 1e-4
    wd = 1e-4
    X = np.ones((n_samples, n_hid)) / n_hid
    y = np.ones((n_samples, 1)) * scale_init

    for dtype in [tf.float16, tf.float32]:
      # pylint: disable=no-member
      regularizer = tf.contrib.layers.l2_regularizer(wd)

      with tf.Graph().as_default() as g:
        x_ph = tf.placeholder(dtype, [n_samples, n_hid])
        y_ph = tf.placeholder(dtype, [n_samples, 1])

        y_pred = tf.layers.dense(
            x_ph, 1, kernel_regularizer=regularizer,
            use_bias=False,
            kernel_initializer=tf.constant_initializer(scale_init, dtype=dtype),
        )
        loss = tf.reduce_mean((y_ph - y_pred) ** 2)
        reg_loss = tf.losses.get_regularization_loss()
        loss += reg_loss
        opt = tf.train.AdamOptimizer()
        grad = opt.compute_gradients(loss)[0][0]

        with self.test_session(g, use_gpu=True) as sess:
          sess.run(tf.global_variables_initializer())
          reg_loss_val, grad_val = sess.run([reg_loss, grad],
                                            {x_ph: X, y_ph: y})
      if dtype == tf.float16:
        self.assertEqual(reg_loss_val, 0.0)
        npt.assert_allclose(grad_val, np.zeros((2, 1), dtype=np.float16))
      else:
        self.assertAlmostEqual(reg_loss_val, 1e-12)
        npt.assert_allclose(grad_val, np.ones((2, 1)) * 1e-8)

  def test_regularization_mixed(self):
    n_samples = 3
    n_hid = 2
    scale_init = 1e-4
    wd = 1e-4
    X = np.ones((n_samples, n_hid)) / n_hid
    y = np.ones((n_samples, 1)) * scale_init

    dtype = tf.float16
    # pylint: disable=no-member
    regularizer = mp_regularizer_wrapper(tf.contrib.layers.l2_regularizer(wd))

    with tf.Graph().as_default() as g:
      x_ph = tf.placeholder(dtype, [n_samples, n_hid])
      y_ph = tf.placeholder(dtype, [n_samples, 1])

      y_pred = tf.layers.dense(
          x_ph, 1, kernel_regularizer=regularizer,
          use_bias=False,
          kernel_initializer=tf.constant_initializer(scale_init, dtype=dtype),
      )
      loss = tf.reduce_mean((y_ph - y_pred) ** 2)
      reg_loss = tf.losses.get_regularization_loss()
      loss += tf.cast(reg_loss, loss.dtype)
      opt = MixedPrecisionOptimizerWrapper(tf.train.AdamOptimizer())
      grad = opt.compute_gradients(loss)[0][0]

      with self.test_session(g, use_gpu=True) as sess:
        sess.run(tf.global_variables_initializer())
        reg_loss_val, grad_val = sess.run([reg_loss, grad],
                                          {x_ph: X, y_ph: y})

    self.assertAlmostEqual(reg_loss_val, 0.0)
    self.assertEqual(reg_loss.name, "Const_1:0")
    npt.assert_allclose(grad_val, np.ones((2, 1)) * 1e-8, atol=1e-11)

  def test_convergence(self):
    for dtype in ['mixed', tf.float32]:
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
        train_op = optimize_loss(loss, "Adam", {},
                                 lambda gs: fixed_lr(gs, 0.05), dtype=dtype)

        with self.test_session(g, use_gpu=True) as sess:
          sess.run(tf.global_variables_initializer())
          for i in range(6000):
            sess.run(train_op, {x_ph: X, y_ph: y})
          w_learned = sess.run(tf.trainable_variables()[0])

        npt.assert_allclose(w_learned, w, atol=0.01)


if __name__ == '__main__':
  tf.test.main()
