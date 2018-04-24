# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from open_seq2seq.decoders import TransformerDecoder
from open_seq2seq.data.text2text import SpecialTextTokens


class TransformerDecoderTest(tf.test.TestCase):
  def setUp(self):
    print("TransformerDecoderTest Test")

  def tearDown(self):
    print("TTransformerDecoderTest Test")

  def test_transformerDecoder(self):
    # this tests checks that after consuming GO symbol, both encoder
    # modes (train and eval) yield equal results
    batch_size = 1
    T = 1
    dim = 512
    tgt_voc_size = 14
    layers = 6

    decoder_params = {
      "initializer": tf.uniform_unit_scaling_initializer,
      "use_encoder_emb": True,
      "tie_emb_and_proj": True,
      "d_model": dim,
      "ffn_inner_dim": dim*2,
      "decoder_layers": layers,
      "attention_heads": 2,
      "decoder_drop_prob": 0.0,
      "batch_size_per_gpu": batch_size,
      "tgt_vocab_size": tgt_voc_size,
      "GO_SYMBOL": SpecialTextTokens.S_ID.value,
      "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
      "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,
      "dtype": tf.float32,
    }

    for r in range(2):
      print("*** R: {}".format(r))
      enc_out = tf.placeholder(dtype=tf.float32, shape=[batch_size, T, dim])
      enc_emb_w = tf.placeholder(dtype=tf.float32, shape=[tgt_voc_size, dim])
      input_sequence = tf.placeholder(dtype=tf.int32, shape=[batch_size, T])
      target_sequence = tf.placeholder(dtype=tf.int32, shape=[batch_size, T])
      tgt_length = tf.constant(value=T, shape=[batch_size])

      encoder_output = {'encoder_outputs': enc_out,
       'encoder_state': None,
       'src_lengths': None,
       'enc_emb_w': enc_emb_w,
       'encoder_input': input_sequence}

      decoder_input = {
          "encoder_output": encoder_output,
          "tgt_inputs": target_sequence,
          "tgt_lengths": tgt_length
        }

      with tf.variable_scope("Decoder_{}".format(r), initializer=tf.uniform_unit_scaling_initializer) as scope:
        tf.set_random_seed(1234)
        decoderT = TransformerDecoder(params=decoder_params, mode="train")
        decoderT_out = decoderT.decode(decoder_input)
        tf.set_random_seed(1234)
        scope.reuse_variables()
        decoderE = TransformerDecoder(params=decoder_params, mode="eval")
        decoderE._is_unittest = True
        decoderE_out = decoderE.decode(decoder_input)


      feed_dict = {enc_out: np.random.random(size=(batch_size, T, dim)),
                   enc_emb_w: np.random.random(size=(tgt_voc_size, dim)),
                   input_sequence: np.random.randint(low=4, high=tgt_voc_size-1,
                                                     size=(batch_size, T)),
                   target_sequence: np.random.randint(low=4, high=tgt_voc_size-1,
                                                      size=(batch_size, T))
                   }

      with self.test_session(use_gpu=True) as sess:
        sess.run(tf.global_variables_initializer())
        edT = sess.run(decoderT_out["decoder_samples"], feed_dict=feed_dict)
        edE = sess.run(decoderE_out["decoder_samples"], feed_dict=feed_dict)
        print(len(edE))
        print(len(edT))
        print("Train: {}".format(edT))
        print("Eval:  {}".format(edE))
        self.assertAllEqual(edT, edE)


