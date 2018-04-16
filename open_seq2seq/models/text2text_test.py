# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import runpy

from open_seq2seq.utils.model_builders import create_encoder_decoder_loss_model
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
      "./example_configs/text2text/toy_task_nmt-reversal.py")
    train_config = config_module['base_params']
    if 'train_params' in config_module:
      train_config.update(config_module['train_params'])

    # TODO: should we maybe have just a single directory parameter?
    train_config['data_layer_params']['src_vocab_file'] = "./toy_data/vocab/source.txt"
    train_config['data_layer_params']['tgt_vocab_file'] = "./toy_data/vocab/target.txt"
    train_config['data_layer_params']['source_file'] = "./toy_data/train/source.txt"
    train_config['data_layer_params']['target_file'] = "./toy_data/train/target.txt"

    with tf.Graph().as_default():
      model = create_encoder_decoder_loss_model(train_config, "train", None)
      with self.test_session(use_gpu=True) as sess:
        tf.global_variables_initializer().run()
        for num in range(0, 2):
          for i, model_dict in enumerate(model.data_layer.iterate_one_epoch()):
            loss, _ = sess.run(
              [model.loss, model.train_op],
              feed_dict=model_dict,
            )


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
      "./example_configs/text2text/toy_task_nmt-reversal.py")
    train_config = config_module['base_params']
    if 'train_params' in config_module:
      train_config.update(config_module['train_params'])

    # TODO: should we maybe have just a single directory parameter?
    train_config['data_layer_params']['src_vocab_file'] = "./toy_data/vocab/source.txt"
    train_config['data_layer_params']['tgt_vocab_file'] = "./toy_data/vocab/target.txt"
    train_config['data_layer_params']['source_file'] = "./toy_data/train/source.txt"
    train_config['data_layer_params']['target_file'] = "./toy_data/train/target.txt"
    train_config["use_horovod"] = True

    with tf.Graph().as_default():
      model = create_encoder_decoder_loss_model(train_config, "train", None)
      with self.test_session(use_gpu=True) as sess:
        tf.global_variables_initializer().run()
        for num in range(0, 2):
          for i, model_dict in enumerate(model.data_layer.iterate_one_epoch()):
            loss, _ = sess.run(
              [model.loss, model.train_op],
              feed_dict=model_dict,
            )


if __name__ == '__main__':
  tf.test.main()
