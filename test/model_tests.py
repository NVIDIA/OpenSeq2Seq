# Copyright (c) 2017 NVIDIA Corporation
import tensorflow as tf
import sys
sys.path.append("data")
sys.path.append("model")
from .context import open_seq2seq
import numpy as np

def create_params():
  params = {}
  # data layer description
  params['src_vocab_file'] = "test/toy_data/vocab/source.txt"
  params['tgt_vocab_file'] = "test/toy_data/vocab/target.txt"
  params['batch_size'] = 64
  params['source_file'] = "test/toy_data/train/source.txt"
  params['target_file'] = "test/toy_data/train/target.txt"
  params["mode"] = "train"
  params['shuffle'] = True
  params['delimiter'] = " "
  params['bucket_src'] = [55]
  params['bucket_tgt'] = [55]
  # model description
  params['encoder_cell_type'] = 'lstm'
  params["encoder_type"] = "unidirectional"
  params['encoder_cell_units'] = 16
  params['encoder_layers'] = 2
  params['encoder_dp_input_keep_prob'] = 1.0
  params['encoder_dp_output_keep_prob'] = 1.0
  params['encoder_use_skip_connections'] = False

  params['decoder_cell_type'] = 'lstm'
  params['decoder_cell_units'] = 16
  params['decoder_layers'] = 2
  params['decoder_dp_input_keep_prob'] = 1.0
  params['decoder_dp_output_keep_prob'] = 1.0
  params['decoder_use_skip_connections'] = False

  params['src_emb_size'] = 16
  params['tgt_emb_size'] = 16
  params["use_attention"] = True
  params['attention_type'] = 'bahdanau'
  params['attention_layer_size'] = 16
  params['optimizer'] = 'Adam'
  params['learning_rate'] = 0.001
  return params

class BasicSeq2SeqWithAttentionTest(tf.test.TestCase):
  def setUp(self):
    print("Setting Up BasicSeq2SeqWithAttention")

  def tearDown(self):
    print("Tear down BasicSeq2SeqWithAttention")

  def test_train(self):
    params = create_params()
    dl = open_seq2seq.data.data_layer.ParallelDataInRamInputLayer(params)
    params['src_vocab_size'] = len(dl.source_seq2idx)
    params['tgt_vocab_size'] = len(dl.target_seq2idx)
    model = open_seq2seq.model.seq2seq_model.BasicSeq2SeqWithAttention(params)

    with self.test_session(use_gpu=True) as sess:
      tf.global_variables_initializer().run()
      for num in range(0, 2):
        t_iters = 0
        for i, (x, y, bucket_id, len_x, len_y) in enumerate(dl.iterate_one_epoch()):
          model_dict = {
            model.x : x[:, 0:np.max(len_x)],
            model.y : y[:, 0:np.max(len_y)],
            model.x_length: len_x,
            model.y_length: len_y
          }
          if i % 50 == 0:
            loss, _, fo = sess.run([model.loss, model.train_op, model._final_outputs], feed_dict=model_dict)
            print("Current LOSS: {}".format(loss))
            ctr = 0
            for row in fo.sample_id:
              print(len_y[ctr])
              print([dl.target_idx2seq[p] for p in y[ctr]])
              print([dl.target_idx2seq[p] for p in row])
              ctr += 1
              print("xxxxxxxxxxxxxxxxxxxxx")
          else:
            loss, _ = sess.run([model.loss, model.train_op], feed_dict=model_dict)
          t_iters += 1

        print("TOTAL ITERATIONS IN EPOCH:")
        print(t_iters)

class BasicSeq2SeqWithAttentionTestOnHorovod(tf.test.TestCase):
  def setUp(self):
    print("Setting Up BasicSeq2SeqWithAttention on Horovod")

  def tearDown(self):
    print("Tear down BasicSeq2SeqWithAttention on Horovod")

  def test_train(self):
    try:
      print("Attempting BasicSeq2SeqWithAttention on Horovod")
      import horovod.tensorflow as hvd
      hvd.init()
      params = create_params()
      dl = open_seq2seq.data.data_layer.ParallelDataInRamInputLayer(params,
                                                                    hvd.size(),
                                                                    hvd.rank())
      params['src_vocab_size'] = len(dl.source_seq2idx)
      params['tgt_vocab_size'] = len(dl.target_seq2idx)
      with tf.variable_scope("Horovod"):
        model = open_seq2seq.model.seq2seq_model.BasicSeq2SeqWithAttention(model_params=params,
                                                                           gpu_ids="horovod")
        with self.test_session(use_gpu=True) as sess:
          tf.global_variables_initializer().run()
          for num in range(0, 1):
            t_iters = 0
            for i, (x, y, bucket_id, len_x, len_y) in enumerate(dl.iterate_one_epoch()):
              model_dict = {
                model.x : x[:, 0:np.max(len_x)],
                model.y : y[:, 0:np.max(len_y)],
                model.x_length: len_x,
                model.y_length: len_y
              }
              if i % 50 == 0:
                loss, _, fo = sess.run([model.loss, model.train_op, model._final_outputs], feed_dict=model_dict)
                print("Current LOSS: {}".format(loss))
                ctr = 0
                for row in fo.sample_id:
                  print(len_y[ctr])
                  print([dl.target_idx2seq[p] for p in y[ctr]])
                  print([dl.target_idx2seq[p] for p in row])
                  ctr += 1
                  print("xxxxxxxxxxxxxxxxxxxxx")
              else:
                loss, _ = sess.run([model.loss, model.train_op], feed_dict=model_dict)
              t_iters += 1

            print("TOTAL ITERATIONS IN EPOCH:")
            print(t_iters)
    except:
      print("Could not test on Horovod. Is it installed?")


