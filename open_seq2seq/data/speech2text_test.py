# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import numpy.testing as npt

from .speech2text import Speech2TextDataLayer
from .data_layer import MultiGPUWrapper


class Speech2TextDataLayerTests(tf.test.TestCase):
  def setUp(self):
    self.params = {
      'batch_size': 2,
      'alphabet_config_path': 'open_seq2seq/test_utils/toy_speech_data/alphabet.txt',
      'dataset_path': [
        'open_seq2seq/test_utils/toy_speech_data/toy_data.csv',
      ],
      'num_audio_features': 161,
      'input_type': 'spectrogram',
    }

  def tearDown(self):
    pass

  def test_init(self):
    pass

  def check_input_tensors(self):
    input_tensors = self.data_layer.get_input_tensors()
    self.assertEqual(len(input_tensors), 4)
    # checking self.x
    self.assertEqual(len(input_tensors[0]), self.num_gpus)
    for tensor in input_tensors[0]:
      self.assertListEqual(
        tensor.shape.as_list(),
        [self.params['batch_size'], None, self.params['num_audio_features']],
      )
      self.assertEqual(tensor.dtype, tf.float32)
    # checking self.x_len
    self.assertEqual(len(input_tensors[1]), self.num_gpus)
    for tensor in input_tensors[1]:
      self.assertListEqual(tensor.shape.as_list(), [self.params['batch_size']])
      self.assertEqual(tensor.dtype, tf.int32)
    # checking self.y
    self.assertEqual(len(input_tensors[2]), self.num_gpus)
    for tensor in input_tensors[2]:
      self.assertListEqual(
        tensor.shape.as_list(),
        [self.params['batch_size'], None],
      )
      self.assertEqual(tensor.dtype, tf.int32)
    # checking self.y_len
    self.assertEqual(len(input_tensors[3]), self.num_gpus)
    for tensor in input_tensors[3]:
      self.assertListEqual(tensor.shape.as_list(), [self.params['batch_size']])
      self.assertEqual(tensor.dtype, tf.int32)

    length = [10, 15, 3, 3]
    sample_x, sample_x_len, sample_y, sample_y_len = [], [], [], []
    for i in range(self.num_gpus):
      sample_x.append(
        np.random.rand(
          self.params['batch_size'],
          length[i],
          self.params['num_audio_features'],
        ).astype(np.float32)
      )
      sample_x_len.append(
        np.random.randint(
          10,
          size=self.params['batch_size'],
          dtype=np.int32,
        )
      )
      sample_y.append(
        np.random.randint(
          10,
          size=(self.params['batch_size'], length[i]),
          dtype=np.int32,
        )
      )
      sample_y_len.append(
        np.random.randint(
          10,
          size=self.params['batch_size'],
          dtype=np.int32,
        )
      )

    with self.test_session(use_gpu=True) as sess:
      input_tensors_vals = sess.run(
        input_tensors,
        feed_dict={
          input_tensors[0]: tuple(sample_x),
          input_tensors[1]: tuple(sample_x_len),
          input_tensors[2]: tuple(sample_y),
          input_tensors[3]: tuple(sample_y_len),
        },
      )
    for i in range(self.num_gpus):
      npt.assert_allclose(input_tensors_vals[0][i], sample_x[i])
    for i in range(self.num_gpus):
      npt.assert_allclose(input_tensors_vals[1][i], sample_x_len[i])
    for i in range(self.num_gpus):
      npt.assert_allclose(input_tensors_vals[2][i], sample_y[i])
    for i in range(self.num_gpus):
      npt.assert_allclose(input_tensors_vals[3][i], sample_y_len[i])

  def test_get_input_tensors_1gpu(self):
    self.num_gpus = 1
    self.data_layer = MultiGPUWrapper(
      Speech2TextDataLayer(self.params),
      self.num_gpus,
    )
    self.check_input_tensors()

  def test_get_input_tensors_2gpus(self):
    self.num_gpus = 2
    self.data_layer = MultiGPUWrapper(
      Speech2TextDataLayer(self.params),
      self.num_gpus,
    )
    self.check_input_tensors()

  def test_get_input_tensors_4gpus(self):
    self.num_gpus = 4
    self.data_layer = MultiGPUWrapper(
      Speech2TextDataLayer(self.params),
      self.num_gpus,
    )
    self.check_input_tensors()

  def test_get_one_sample(self):
    pass

  def test_get_one_batch(self):
    pass

  def test_get_feed_dict_per_gpu(self):
    pass

  def test_iterate_one_epoch(self):
    pass

  def test_iterate_forever(self):
    pass


if __name__ == '__main__':
  tf.test.main()
