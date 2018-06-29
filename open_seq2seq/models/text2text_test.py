# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
import runpy

from open_seq2seq.test_utils.create_reversed_examples import create_data, \
                                                             remove_data


class BasicText2TextWithAttentionTest(tf.test.TestCase):
  def setUp(self):
    print("Setting Up BasicSeq2SeqWithAttention")
    create_data(train_corpus_size=500)

  def tearDown(self):
    print("Tear down BasicSeq2SeqWithAttention")
    remove_data()

  def test_train(self):
    config_module = runpy.run_path(
      "./example_configs/text2text/nmt-reversal-RR.py")
    train_config = config_module['base_params']
    if 'train_params' in config_module:
      train_config.update(config_module['train_params'])

    # TODO: should we maybe have just a single directory parameter?
    train_config['data_layer_params']['src_vocab_file'] = "./toy_data/vocab/source.txt"
    train_config['data_layer_params']['tgt_vocab_file'] = "./toy_data/vocab/target.txt"
    train_config['data_layer_params']['source_file'] = "./toy_data/train/source.txt"
    train_config['data_layer_params']['target_file'] = "./toy_data/train/target.txt"

    step = 0
    with tf.Graph().as_default():
      model = config_module['base_model'](train_config, "train", None)
      model.compile()
      with self.test_session(use_gpu=True) as sess:
        tf.global_variables_initializer().run()
        sess.run(model.get_data_layer().iterator.initializer)
        while True:
          try:
            loss, _ = sess.run(
              [model.loss, model.train_op]
            )
          except tf.errors.OutOfRangeError:
            break
          step += 1
          if step >=25:
            break


class BasicText2TextWithAttentionTestOnHorovod(tf.test.TestCase):
  def setUp(self):
    print("Setting Up BasicSeq2SeqWithAttention on Horovod")
    create_data(train_corpus_size=500)

  def tearDown(self):
    print("Tear down BasicSeq2SeqWithAttention on Horovod")
    remove_data()

  def test_train(self):
    try:
      import horovod.tensorflow as hvd
    except ImportError:
      print("Could not test on Horovod. Is it installed?")
      return

    print("Attempting BasicSeq2SeqWithAttention on Horovod")
    hvd.init()
    config_module = runpy.run_path(
      "./example_configs/text2text/toy-reversal/nmt-reversal-RR.py")
    train_config = config_module['base_params']
    if 'train_params' in config_module:
      train_config.update(config_module['train_params'])

    # TODO: should we maybe have just a single directory parameter?
    train_config['data_layer_params']['src_vocab_file'] = "./toy_data/vocab/source.txt"
    train_config['data_layer_params']['tgt_vocab_file'] = "./toy_data/vocab/target.txt"
    train_config['data_layer_params']['source_file'] = "./toy_data/train/source.txt"
    train_config['data_layer_params']['target_file'] = "./toy_data/train/target.txt"
    train_config["use_horovod"] = True
    #train_config['data_layer_params']["repeat"] = False
    step = 0
    with tf.Graph().as_default():
      model = config_module['base_model'](train_config, "train", None)
      model.compile()
      with self.test_session(use_gpu=True) as sess:
        tf.global_variables_initializer().run()
        sess.run(model.get_data_layer().iterator.initializer)
        while True:
          try:
            loss, _ = sess.run(
              [model.loss, model.train_op]
            )
          except tf.errors.OutOfRangeError:
            break
          step += 1
          if step >=25:
            break

if __name__ == '__main__':
  tf.test.main()
