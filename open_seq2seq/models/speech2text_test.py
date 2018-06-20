# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
import numpy as np
import copy
import numpy.testing as npt
import tempfile
import os
import pandas as pd

from .speech2text import levenshtein
from open_seq2seq.test_utils.test_speech_config import base_params, \
                                                       train_params, \
                                                       eval_params, \
                                                       base_model
from open_seq2seq.utils import train, evaluate, infer
from open_seq2seq.utils.utils import get_available_gpus


class Speech2TextModelTests(tf.test.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def run_model(self, train_config, eval_config, hvd=None):
    with tf.Graph().as_default() as g:
      train_model = base_model(params=train_config, mode="train", hvd=hvd)
      train_model.compile()
      eval_model = base_model(params=eval_config, mode="eval", hvd=hvd)
      eval_model.compile(force_var_reuse=True)

      train(train_model, eval_model)
      saver = tf.train.Saver()
      checkpoint = tf.train.latest_checkpoint(train_model.params['logdir'])
      with self.test_session(g, use_gpu=True) as sess:
        saver.restore(sess, checkpoint)
        sess.run([train_model.get_data_layer(i).iterator.initializer
                  for i in range(train_model.num_gpus)])
        sess.run([eval_model.get_data_layer(i).iterator.initializer
                  for i in range(eval_model.num_gpus)])

        weights = sess.run(tf.trainable_variables())
        loss = sess.run(train_model.loss)
        eval_losses = sess.run(eval_model.eval_losses)
        eval_loss = np.mean(eval_losses)
        weights_new = sess.run(tf.trainable_variables())

        # checking that the weights has not changed from just computing the loss
        for w, w_new in zip(weights, weights_new):
          npt.assert_allclose(w, w_new)
      eval_dict = evaluate(eval_model, checkpoint)
    return loss, eval_loss, eval_dict

  def prepare_config(self):
    base_params['logdir'] = tempfile.mktemp()
    train_config = copy.deepcopy(base_params)
    eval_config = copy.deepcopy(base_params)
    train_config.update(copy.deepcopy(train_params))
    eval_config.update(copy.deepcopy(eval_params))
    return train_config, eval_config

  def test_regularizer(self):
    for dtype in [tf.float16, tf.float32, 'mixed']:
      train_config, eval_config = self.prepare_config()
      train_config['num_epochs'] = 60
      train_config.update({
        "dtype": dtype,
        "regularizer": tf.contrib.layers.l2_regularizer,
        "regularizer_params": {
          'scale': 1e4,
        },
      })
      eval_config.update({
        "dtype": dtype,
      })
      loss, eval_loss, eval_dict = self.run_model(train_config, eval_config)

      self.assertGreaterEqual(loss, 500.0)
      self.assertGreaterEqual(eval_loss, 500.0)
      self.assertGreaterEqual(eval_dict['Eval WER'], 0.95)

  def test_convergence(self):
    for dtype in [tf.float32, "mixed"]:
      train_config, eval_config = self.prepare_config()
      train_config.update({
        "dtype": dtype,
      })
      eval_config.update({
        "dtype": dtype,
      })
      loss, eval_loss, eval_dict = self.run_model(train_config, eval_config)

      self.assertLess(loss, 5.0)
      self.assertLess(eval_loss, 200.0)
      self.assertLess(eval_dict['Eval WER'], 0.1)

  def test_convergence_with_iter_size(self):
    try:
      import horovod.tensorflow as hvd
      hvd.init()
    except ImportError:
      print("Horovod not installed skipping test_convergence_with_iter_size")
      return

    for dtype in [tf.float32, "mixed"]:
      train_config, eval_config = self.prepare_config()
      train_config.update({
        "dtype": dtype,
        "iter_size": 5,
        "batch_size_per_gpu": 2,
        "use_horovod": True,
      })
      eval_config.update({
        "dtype": dtype,
        "iter_size": 5,
        "batch_size_per_gpu": 2,
        "use_horovod": True,
      })
      loss, eval_loss, eval_dict = self.run_model(train_config, eval_config, hvd)

      self.assertLess(loss, 5.0)
      self.assertLess(eval_loss, 200.0)
      self.assertLess(eval_dict['Eval WER'], 0.1)

  def test_infer(self):
    train_config, infer_config = self.prepare_config()
    train_config['num_epochs'] = 200
    infer_config['batch_size_per_gpu'] = 4

    with tf.Graph().as_default() as g:
      with self.test_session(g, use_gpu=True) as sess:
        gpus = get_available_gpus()

    if len(gpus) > 1:
      infer_config['num_gpus'] = 2
    else:
      infer_config['num_gpus'] = 1

    with tf.Graph().as_default():
      train_model = base_model(params=train_config, mode="train", hvd=None)
      train_model.compile()
      train(train_model, None)

    with tf.Graph().as_default():
      infer_model = base_model(params=infer_config, mode="infer", hvd=None)
      infer_model.compile()

      print(train_model.params['logdir'])
      output_file = os.path.join(train_model.params['logdir'], 'infer_out.csv')
      infer(
        infer_model,
        tf.train.latest_checkpoint(train_model.params['logdir']),
        output_file,
      )
      pred_csv = pd.read_csv(output_file)
      true_csv = pd.read_csv(
        'open_seq2seq/test_utils/toy_speech_data/toy_data.csv',
      )
      for pred_row, true_row in zip(pred_csv.as_matrix(), true_csv.as_matrix()):
        # checking file name
        self.assertEqual(pred_row[0], true_row[0])
        # checking prediction
        self.assertEqual(pred_row[-1], true_row[-1])

  def test_mp_collection(self):
    train_config, eval_config = self.prepare_config()
    train_config['dtype'] = 'mixed'

    with tf.Graph().as_default():
      model = base_model(params=train_config, mode="train", hvd=None)
      model.compile()
      self.assertEqual(len(tf.trainable_variables()), 14)
      self.assertEqual(
        len(tf.get_collection('FP32_MASTER_COPIES')),
        7,  # minus batch norm beta and gamma and row_conv vars
      )

  def test_levenshtein(self):
    s1 = 'this is a great day'
    s2 = 'this is great day'
    self.assertEqual(levenshtein(s1.split(), s2.split()), 1)
    self.assertEqual(levenshtein(s2.split(), s1.split()), 1)
    s1 = 'this is a great day'
    s2 = 'this great day'
    self.assertEqual(levenshtein(s1.split(), s2.split()), 2)
    self.assertEqual(levenshtein(s2.split(), s1.split()), 2)
    s1 = 'this is a great day'
    s2 = 'this great day'
    self.assertEqual(levenshtein(s1.split(), s2.split()), 2)
    self.assertEqual(levenshtein(s2.split(), s1.split()), 2)
    s1 = 'this is a great day'
    s2 = 'this day is a great'
    self.assertEqual(levenshtein(s1.split(), s2.split()), 2)
    self.assertEqual(levenshtein(s2.split(), s1.split()), 2)
    s1 = 'this is a great day'
    s2 = 'this day is great'
    self.assertEqual(levenshtein(s1.split(), s2.split()), 3)
    self.assertEqual(levenshtein(s2.split(), s1.split()), 3)

    s1 = 'london is the capital of great britain'
    s2 = 'london capital gret britain'
    self.assertEqual(levenshtein(s1.split(), s2.split()), 4)
    self.assertEqual(levenshtein(s2.split(), s1.split()), 4)
    self.assertEqual(levenshtein(s1, s2), 11)
    self.assertEqual(levenshtein(s2, s1), 11)

  def test_maybe_functions(self):
    train_config, eval_config = self.prepare_config()

    with tf.Graph().as_default():
      model = base_model(params=train_config, mode="train", hvd=None)
      model.compile()
    model._gpu_ids = range(5)
    model.params['batch_size_per_gpu'] = 2
    char2idx = model.get_data_layer().params['char2idx']
    inputs = [
      ['this is a great day', 'london is the capital of great britain'],
      ['ooo', 'lll'],
      ['a b c\' asdf', 'blah blah bblah'],
      ['this is great day', 'london capital gret britain'],
      ['aaaaaaaasdfdasdf', 'df d sdf asd fd f sdf df blah\' blah'],
    ]
    outputs = [
      ['this is great a day', 'london capital gret britain'],
      ['ooo', 'lll'],
      ['aaaaaaaasdfdasdf', 'df d sdf asd fd f sdf df blah blah'],
      ['this is a great day', 'london is the capital of great britain'],
      ['a b c\' asdf', 'blah blah\' bblah'],
    ]
    y = [None] * len(inputs)
    len_y = [None] * len(inputs)
    indices, values, dense_shape = [], [], []

    num_gpus = len(inputs)
    for gpu_id in range(num_gpus):
      num_samples = len(inputs[gpu_id])
      max_len = np.max(list(map(len, inputs[gpu_id])))
      y[gpu_id] = np.zeros((num_samples, max_len), dtype=np.int)
      len_y[gpu_id] = np.zeros(num_samples, dtype=np.int)
      for sample_id in range(num_samples):
        num_letters = len(inputs[gpu_id][sample_id])
        len_y[gpu_id][sample_id] = num_letters
        for letter_id in range(num_letters):
          y[gpu_id][sample_id, letter_id] = char2idx[
            inputs[gpu_id][sample_id][letter_id]
          ]

    num_gpus = len(outputs)
    for gpu_id in range(num_gpus):
      num_samples = len(outputs[gpu_id])
      max_len = np.max(list(map(len, outputs[gpu_id])))
      dense_shape.append(np.array((num_samples, max_len)))
      values.append([])
      indices.append([])
      for sample_id in range(num_samples):
        num_letters = len(outputs[gpu_id][sample_id])
        for letter_id in range(num_letters):
          values[gpu_id].append(
            char2idx[outputs[gpu_id][sample_id][letter_id]]
          )
          indices[gpu_id].append(np.array([sample_id, letter_id]))
      values[gpu_id] = np.array(values[gpu_id], dtype=np.int)
      indices[gpu_id] = np.array(indices[gpu_id], dtype=np.int)

    x = [np.empty(2)] * len(y)
    len_x = [None] * len(y)
    input_values = list(zip(x, len_x, y, len_y))
    output_values = [
      [tf.SparseTensorValue(indices[i], values[i], dense_shape[i])]
      for i in range(num_gpus)
    ]

    results = []
    for inp, out in zip(input_values, output_values):
      inp_dict = {'source_tensors': [inp[0], inp[1]],
                  'target_tensors': [inp[2], inp[3]]}
      results.append(model.evaluate(inp_dict, out))
    for inp, out in zip(input_values, output_values):
      inp_dict = {'source_tensors': [inp[0], inp[1]],
                  'target_tensors': [inp[2], inp[3]]}
      results.append(model.evaluate(inp_dict, out))
    output_dict = model.finalize_evaluation(results)

    w_lev = 0.0
    w_len = 0.0
    for batch_id in range(len(inputs)):
      for sample_id in range(len(inputs[batch_id])):
        input_sample = inputs[batch_id][sample_id]
        output_sample = outputs[batch_id][sample_id]
        w_lev += levenshtein(input_sample.split(), output_sample.split())
        w_len += len(input_sample.split())

    self.assertEqual(output_dict['Eval WER'], w_lev / w_len)
    self.assertEqual(output_dict['Eval WER'], 37 / 40.0)

    inp_dict = {'source_tensors': [input_values[0][0], input_values[0][1]],
                'target_tensors': [input_values[0][2], input_values[0][3]]}
    output_dict = model.maybe_print_logs(inp_dict, output_values[0])
    self.assertEqual(output_dict['Sample WER'], 0.4)


if __name__ == '__main__':
  tf.test.main()
