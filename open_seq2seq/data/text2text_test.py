# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
import numpy as np

from .text2text import ParallelDataInRamInputLayer, ParallelTextDataLayer
from .data_layer import MultiGPUWrapper
from open_seq2seq.test_utils.create_reversed_examples import create_data, \
                                                             remove_data


class ParallelDataInRamInputLayerTests(tf.test.TestCase):
  def setUp(self):
    create_data()
    # defining some general parameters for all tests
    self.params = {
      'src_vocab_file': "./toy_data/vocab/source.txt",
      'tgt_vocab_file': "./toy_data/vocab/target.txt",
      'source_file': "./toy_data/train/source.txt",
      'target_file': "./toy_data/train/target.txt",
      'shuffle': True,
      'delimiter': " ",
      'pad_vocab_to_eight': True,
    }

  def tearDown(self):
    remove_data()

  def test_load_toy_word_based_data(self):
    self.params.update({
      'batch_size': 4,
      'bucket_src': [10, 20, 50, 100],
      'bucket_tgt': [10, 20, 50, 100],
    })
    num_gpus = 2

    dl = MultiGPUWrapper(
      ParallelDataInRamInputLayer(self.params, None),
      num_gpus,
    )
    dl.build_graph()

    self.assertEqual(dl.params['batch_size'],
                     self.params['batch_size'] * num_gpus)
    self.assertEqual(
      dl.params['target_seq2idx'],
      dl.params['source_seq2idx'],
      msg="Toy source and target vocabs should be the same",
    )
    self.assertEqual(
      dl.params['source_idx2seq'],
      dl.params['target_idx2seq'],
      msg="Error in forming idx2seq for Toy example",
    )

    for feed_dict in dl.iterate_one_epoch():
      # this is supposed to reuse the same session
      with self.test_session(use_gpu=True) as sess:
        x, len_x, y, len_y = sess.run(dl.get_input_tensors(), feed_dict)

      self.assertEqual(len(x), len(y))
      for i in range(len(x)):
        self.assertEqual(x[i].shape[0], y[i].shape[0])
      if type(x) is not np.ndarray:
        self.assertEqual(len(x), num_gpus)
        for i in range(len(x)):
          self.assertEqual(x[i].shape[1], y[i].shape[1])
          self.assertEqual(x[i].shape[0], self.params['batch_size'])
      else:
        self.assertEqual(x.shape[0], self.params['batch_size'])

  def test_nmt_data_layer_bucket_sizer(self):
    self.params.update({
      'batch_size': 16,
      'bucket_src': [10, 20, 50, 100],
      'bucket_tgt': [10, 15, 30, 100],
    })
    num_gpus = 2

    dl = MultiGPUWrapper(
      ParallelDataInRamInputLayer(self.params, None),
      num_gpus,
    )
    dl.build_graph()

    for i in range(0, 11):
      self.assertEqual(
        dl._data_layer.determine_bucket(i, self.params['bucket_src']), 0)
    for i in range(11, 21):
      self.assertEqual(
        dl._data_layer.determine_bucket(i, self.params['bucket_src']), 1)
    for i in range(21, 51):
      self.assertEqual(
        dl._data_layer.determine_bucket(i, self.params['bucket_src']), 2)
    for i in range(51, 101):
      self.assertEqual(
        dl._data_layer.determine_bucket(i, self.params['bucket_src']), 3)

    self.assertEqual(
      dl._data_layer.determine_bucket(310, self.params['bucket_src']),
      dl._data_layer.OUT_OF_BUCKET,
    )

    for bucket_id in dl._data_layer._bucket_sizes.keys():
      self.assertEqual(
        dl._data_layer._bucket_sizes[bucket_id],
        dl._data_layer._bucket_id_to_src_example[bucket_id].shape[0],
      )
      self.assertEqual(
        dl._data_layer._bucket_sizes[bucket_id],
        dl._data_layer._bucket_id_to_tgt_example[bucket_id].shape[0],
      )

  def check_one_epoch(self, dl, num_gpus):
    for feed_dict in dl.iterate_one_epoch():
      # this is supposed to reuse the same session
      with self.test_session(use_gpu=True) as sess:
        x, len_x, y, len_y = sess.run(dl.get_input_tensors(), feed_dict)

      self.assertEqual(len(x), len(y))
      for i in range(len(x)):
        self.assertEqual(x[i].shape[0], y[i].shape[0])
      if type(x) is not np.ndarray:
        self.assertEqual(len(x), num_gpus)
        for i in range(len(x)):
          self.assertEqual(x[i].shape[1], y[i].shape[1])
          self.assertEqual(x[i].shape[0], self.params['batch_size'])
      else:
        self.assertEqual(x.shape[0], self.params['batch_size'])


  def test_load_toy_word_based_data_dist(self):
    self.params.update({
      'batch_size': 4,
      'bucket_src': [100],
      'bucket_tgt': [100],
    })

    num_workers = 13
    num_gpus = 2

    dl_reg = MultiGPUWrapper(
      ParallelDataInRamInputLayer(self.params, None),
      num_gpus,
    )
    dl_reg.build_graph()
    dls_hvd = [
      ParallelDataInRamInputLayer(
        self.params, None, num_workers=1, worker_id=0,
      )
    ]
    dls_hvd[-1].build_graph()
    for ind in range(num_workers):
      dls_hvd.append(ParallelDataInRamInputLayer(
        self.params, None,
        num_workers=num_workers,
        worker_id=ind,
      ))
      dls_hvd[-1].build_graph()

    self.assertEqual(dl_reg.params['batch_size'],
                     self.params['batch_size'] * num_gpus)
    self.assertEqual(
      dl_reg.params['target_seq2idx'],
      dl_reg.params['source_seq2idx'],
      msg="Toy source and target vocabs should be the same",
    )
    self.assertEqual(
      dl_reg.params['source_idx2seq'],
      dl_reg.params['target_idx2seq'],
      msg="Error in forming idx2seq for Toy example",
    )
    self.assertEqual(
      len(dl_reg._data_layer._bucket_id_to_src_example[0]),
      len(dl_reg._data_layer._bucket_id_to_tgt_example[0]),
    )
    self.assertNotEqual(len(dl_reg._data_layer._bucket_id_to_src_example[0]), 0)
    self.check_one_epoch(dl_reg, num_gpus)

    for q, dl in enumerate(dls_hvd):
      self.assertEqual(dl.params['batch_size'],
                       self.params['batch_size'])
      self.assertEqual(
        dl.params['target_seq2idx'],
        dl.params['source_seq2idx'],
        msg="Toy source and target vocabs should be the same",
      )
      self.assertEqual(
        dl.params['source_idx2seq'],
        dl.params['target_idx2seq'],
        msg="Error in forming idx2seq for Toy example",
      )
      self.assertEqual(
        len(dl._bucket_id_to_src_example[0]),
        len(dl._bucket_id_to_tgt_example[0]),
      )
      self.assertNotEqual(len(dl._bucket_id_to_src_example[0]), 0)
      self.check_one_epoch(dl, num_gpus)

    total = len(dl_reg._data_layer._bucket_id_to_src_example[0])
    self.assertEqual(total, len(dls_hvd[0]._bucket_id_to_src_example[0]))
    cnt = 0
    for dl in dls_hvd[1:]:
      cnt += len(dl._bucket_id_to_src_example[0])
    print('total size:')
    print(total)
    self.assertEqual(cnt, total)


class ParallelTextDataLayerTests(tf.test.TestCase):
  def setUp(self):
    create_data()
    batch_size = 2
    self.params = {
      'src_vocab_file': "./toy_data/vocab/source.txt",
      'tgt_vocab_file': "./toy_data/vocab/target.txt",
      'source_file': "./toy_data/train/source.txt",
      'target_file': "./toy_data/train/target.txt",
      'shuffle': True,
      'batch_size': batch_size,
      'max_length': 56,
      'repeat': False,
      'delimiter': ' ',
      'map_parallel_calls': 1,
      'prefetch_buffer_size': 1,
    }

  def tearDown(self):
    remove_data()

  def test_init_test4(self):
    dl = ParallelTextDataLayer(params=self.params, model=None)
    dl.build_graph()
    print(len(dl.src_seq2idx))
    print(len(dl.tgt_seq2idx))
    with self.test_session(use_gpu=True) as sess:
      et = sess.run(dl.get_input_tensors())
      self.assertEqual(len(et), 4)
      self.assertEqual(et[0].shape[0], self.params['batch_size'])
      self.assertLessEqual(et[0].shape[1], self.params['max_length'])
      self.assertEqual(et[1].shape[0], self.params['batch_size'])
      self.assertEqual(et[2].shape[0], self.params['batch_size'])
      self.assertLessEqual(et[2].shape[1], self.params['max_length'])
      self.assertEqual(et[3].shape[0], self.params['batch_size'])

  def test_init_test2(self):
    self.params['use_targets'] = False # in this case we do not yield targets
    self.params['shuffle'] = False  # in this case we do not yield targets
    dl = ParallelTextDataLayer(params=self.params, model=None)
    dl.build_graph()
    print(len(dl.src_seq2idx))
    print(len(dl.tgt_seq2idx))
    with self.test_session(use_gpu=True) as sess:
      et = sess.run(dl.get_input_tensors())
      self.assertEqual(len(et), 2)
      self.assertEqual(et[0].shape[0], self.params['batch_size'])
      self.assertLessEqual(et[0].shape[1], self.params['max_length'])
      self.assertEqual(et[1].shape[0], self.params['batch_size'])

  def test_pad8(self):
    self.params['shuffle'] = False  # in this case we do not yield targets
    self.params['pad_lengths_to_eight'] = True
    dl = ParallelTextDataLayer(params=self.params, model=None)
    dl.build_graph()
    print(len(dl.src_seq2idx))
    print(len(dl.tgt_seq2idx))
    with self.test_session(use_gpu=True) as sess:
      et = sess.run(dl.get_input_tensors())
      self.assertEqual(len(et), 4)
      self.assertEqual(et[0].shape[0], self.params['batch_size'])
      self.assertTrue(et[0].shape[1] % 8 == 0)
      self.assertEqual(et[1].shape[0], self.params['batch_size'])
      self.assertEqual(et[2].shape[0], self.params['batch_size'])
      self.assertTrue(et[2].shape[1] % 8 == 0)
      self.assertEqual(et[3].shape[0], self.params['batch_size'])


if __name__ == '__main__':
  tf.test.main()
