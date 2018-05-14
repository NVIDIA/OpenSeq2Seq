# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
import tensorflow as tf
from open_seq2seq.data.text2text import ParallelTextDataLayer, TransformerDataLayer
from open_seq2seq.test_utils.create_reversed_examples import create_data, \
                                                             remove_data


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
      'mode': 'train',
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


class TransformerDataLayerTests(tf.test.TestCase):
  def setUp(self):
    create_data()
    batch_size = 512
    self.params = {
      'data_dir': "/home/okuchaiev/repos/forks/reference/translation/processed_data/",
      'file_pattern': "*dev*",
      'src_vocab_file': "/home/okuchaiev/repos/forks/reference/translation/processed_data/vocab.ende.32768",
      'batch_size': batch_size,
      'max_length': 256,
      'shuffle': True,
      'repeat': 1,
      'mode' : 'train',
    }

  def test_TransformerDataLayer(self):
    print("####################################################")
    print("# --------------------------------------------------")
    print("# -- Starting transformer data layer test ----------")
    print("# --------------------------------------------------")
    dl = TransformerDataLayer(params=self.params, model=None)
    dl.build_graph()
    print(len(dl.src_seq2idx))
    print(len(dl.tgt_seq2idx))

    #iterator = dl.get_dataset_object().make_one_shot_iterator()
    #x, y = iterator.get_next()
    #len_x = tf.count_nonzero(x, axis=1)
    #len_y = tf.count_nonzero(y, axis=1)
    iterator = dl.iterator
    inputs = dl.get_input_tensors()
    #inputs1 = dl.gen_input_tensors()

    with self.test_session(use_gpu=True) as sess:
      print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
      sess.run(iterator.initializer)
      while True:
        try:
          ex, elen_x, ey, elen_y = sess.run(inputs)
          print(ex.shape)
          print(elen_x.shape)
          print(ey.shape)
          print(elen_y.shape)
        except tf.errors.OutOfRangeError:
          break
      print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
      #dl.redo_iterator()
      sess.run(iterator.initializer)
      while True:
        try:
          ex, elen_x, ey, elen_y = sess.run(inputs)
          print(ex.shape)
          print(elen_x.shape)
          print(ey.shape)
          print(elen_y.shape)
        except tf.errors.OutOfRangeError:
          break

if __name__ == '__main__':
  tf.test.main()