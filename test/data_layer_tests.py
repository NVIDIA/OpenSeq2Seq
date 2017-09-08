# Copyright (c) 2017 NVIDIA Corporation
import unittest
from .context import open_seq2seq

class ParallelDataInRamInputLayerTests(unittest.TestCase):

  def test_load_toy_word_based_data(self):
    params = {}
    params['src_vocab_file'] = "test/toy_data/vocab/source.txt"
    params['tgt_vocab_file'] = "test/toy_data/vocab/target.txt"
    params['batch_size'] = 4
    params['source_file'] = "test/toy_data/train/source.txt"
    params['target_file'] = "test/toy_data/train/target.txt"
    params["mode"] = "train"
    params['shuffle'] = True
    params['delimiter'] = " "
    params['bucket_src'] = [10, 20, 50, 100]
    params['bucket_tgt'] = [10, 20, 50, 100]
    """
    params['create_vocab']
    params['tokenize_corpus']
    params['char_mode']"""
    dl = open_seq2seq.data.data_layer.ParallelDataInRamInputLayer(params)
    self.assertEqual(dl.batch_size, params['batch_size'])
    self.assertEqual(dl.target_seq2idx, dl.source_seq2idx, msg="Toy source and target vocabs should be the same")
    self.assertEqual(dl.source_idx2seq, dl.target_idx2seq, msg="Error in forming idx2seq for Toy example")

    num_epochs = 1
    for x, y, bucket_id_to_yield, len_x, len_y in dl.iterate_n_epochs(num_epochs):
      self.assertEqual(x.shape[0], y.shape[0])
      self.assertEqual(x.shape[1], y.shape[1])
      self.assertEqual(x.shape[0], params['batch_size'])


  def test_nmt_data_layer_bucket_sizer(self):
      params = {'source_file': 'test/toy_data/train/source.txt',
                'target_file': 'test/toy_data/train/target.txt',
                'bucket_src': [10, 20, 50, 100],
                'bucket_tgt': [10, 15, 30, 100], 'shuffle': True}
      params['src_vocab_file'] = "test/toy_data/vocab/source.txt"
      params['tgt_vocab_file'] = "test/toy_data/vocab/target.txt"
      params['batch_size'] = 16
      params["mode"] = "train"
      params['delimiter'] = " "

      dl = open_seq2seq.data.data_layer.ParallelDataInRamInputLayer(params)
      for i in range(0, 11):
          self.assertEqual(dl.determine_bucket(i, params['bucket_src']), 0)
      for i in range(11, 21):
          self.assertEqual(dl.determine_bucket(i, params['bucket_src']), 1)
      for i in range(21, 51):
          self.assertEqual(dl.determine_bucket(i, params['bucket_src']), 2)
      for i in range(51, 101):
          self.assertEqual(dl.determine_bucket(i, params['bucket_src']), 3)

      self.assertEqual(dl.determine_bucket(310, params['bucket_src']), dl.OUT_OF_BUCKET)

      for bucket_id in dl._bucket_sizes.keys():
          self.assertEqual(dl._bucket_sizes[bucket_id], dl._bucket_id_to_src_example[bucket_id].shape[0])
          self.assertEqual(dl._bucket_sizes[bucket_id], dl._bucket_id_to_tgt_example[bucket_id].shape[0])

if __name__ == '__main__':
    unittest.main()
