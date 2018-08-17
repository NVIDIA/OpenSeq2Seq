# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from open_seq2seq.data.text2text.text2text import ParallelTextDataLayer
from open_seq2seq.test_utils.create_reversed_examples import create_data, \
                                                             remove_data


class ParallelTextDataLayerTests(tf.test.TestCase):
  def setUp(self):
    create_data(train_corpus_size=1000, data_path="tmp1")
    batch_size = 2
    self.params = {
        'src_vocab_file': "tmp1/vocab/source.txt",
        'tgt_vocab_file': "tmp1/vocab/target.txt",
        'source_file': "tmp1/train/source.txt",
        'target_file': "tmp1/train/target.txt",
        'shuffle': True,
        'batch_size': batch_size,
        'max_length': 56,
        'repeat': False,
        'delimiter': ' ',
        'map_parallel_calls': 1,
        'prefetch_buffer_size': 1,
        'mode': 'train',
    }

  def tearDown(self):
    remove_data(data_path='tmp1')

  def test_init_test4(self):
    dl = ParallelTextDataLayer(params=self.params, model=None)
    dl.build_graph()
    print(len(dl.src_seq2idx))
    print(len(dl.tgt_seq2idx))
    with self.test_session(use_gpu=True) as sess:
      sess.run(dl.iterator.initializer)
      et = sess.run(dl.input_tensors)
      self.assertIn('source_tensors', et)
      self.assertIn('target_tensors', et)
      self.assertEqual(et['source_tensors'][0].shape[0],
                       self.params['batch_size'])
      self.assertLessEqual(et['source_tensors'][0].shape[1],
                           self.params['max_length'])
      self.assertEqual(et['source_tensors'][1].shape[0],
                       self.params['batch_size'])
      self.assertEqual(et['target_tensors'][0].shape[0],
                       self.params['batch_size'])
      self.assertLessEqual(et['target_tensors'][0].shape[1],
                           self.params['max_length'])
      self.assertEqual(et['target_tensors'][1].shape[0],
                       self.params['batch_size'])

  def test_init_test2(self):
    self.params['mode'] = "infer"  # in this case we do not yield targets
    self.params['shuffle'] = False  # in this case we do not yield targets
    dl = ParallelTextDataLayer(params=self.params, model=None)
    dl.build_graph()
    print(len(dl.src_seq2idx))
    print(len(dl.tgt_seq2idx))
    with self.test_session(use_gpu=True) as sess:
      sess.run(dl.iterator.initializer)
      et = sess.run(dl.input_tensors)
      self.assertIn('source_tensors', et)
      self.assertEqual(et['source_tensors'][0].shape[0],
                       self.params['batch_size'])
      self.assertLessEqual(et['source_tensors'][0].shape[1],
                           self.params['max_length'])
      self.assertEqual(et['source_tensors'][1].shape[0],
                       self.params['batch_size'])

  def test_pad8(self):
    self.params['shuffle'] = False  # in this case we do not yield targets
    self.params['pad_lengths_to_eight'] = True
    dl = ParallelTextDataLayer(params=self.params, model=None)
    dl.build_graph()
    print(len(dl.src_seq2idx))
    print(len(dl.tgt_seq2idx))
    print(dl.src_seq2idx)
    print(dl.src_idx2seq)
    for i in range(len(dl.src_seq2idx)):
      self.assertIn(i, dl.src_idx2seq)
    with self.test_session(use_gpu=True) as sess:
      sess.run(dl.iterator.initializer)
      et = sess.run(dl.input_tensors)
      self.assertIn('source_tensors', et)
      self.assertIn('target_tensors', et)
      self.assertEqual(et['source_tensors'][0].shape[0],
                       self.params['batch_size'])
      self.assertTrue(et['source_tensors'][0].shape[1] % 8 == 0)
      self.assertEqual(et['source_tensors'][1].shape[0],
                       self.params['batch_size'])
      self.assertEqual(et['target_tensors'][0].shape[0],
                       self.params['batch_size'])
      self.assertTrue(et['target_tensors'][0].shape[1] % 8 == 0)
      self.assertEqual(et['target_tensors'][1].shape[0],
                       self.params['batch_size'])


if __name__ == '__main__':
  tf.test.main()
