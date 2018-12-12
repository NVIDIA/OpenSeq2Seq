import numpy as np
import pickle
import tensorflow as tf


def load_test_sample(pickle_file):
  with open(pickle_file, 'rb') as f:
    seq, label = pickle.load(f, encoding='bytes')
  return seq, label 

def load_vocab(vocab_file):
  vocab = []
  with open(vocab_file, 'r') as f:
    for line in f:
      vocab.append(line[0])
  vocab.append('_')
  return vocab


class CTCCustomDecoderTests(tf.test.TestCase):

  def setUp(self):
    self.seq, self.label = load_test_sample('ctc_decoder_with_lm/ctc-test.pickle')
    self.vocab = load_vocab('open_seq2seq/test_utils/toy_speech_data/vocab.txt')
    self.beam_width = 16
    self.tol = 1e-3


  def test_decoders(self):
    '''
    Test all CTC decoders on a sample transcript ('ten seconds').
    Standard TF decoders should output 'then seconds'.
    Custom CTC decoder with LM rescoring should yield 'ten seconds'.
    '''
    logits = tf.constant(self.seq)
    seq_len = tf.constant([self.seq.shape[0]])

    greedy_decoded = tf.nn.ctc_greedy_decoder(logits, seq_len, 
        merge_repeated=True)

    beam_search_decoded = tf.nn.ctc_beam_search_decoder(logits, seq_len, 
        beam_width=self.beam_width, 
        top_paths=1, 
        merge_repeated=False)

    custom_op_module = tf.load_op_library('ctc_decoder_with_lm/libctc_decoder_with_kenlm.so')
    decoded_ixs, decoded_vals, decoded_shapes, log_probabilities = (
        custom_op_module.ctc_beam_search_decoder_with_lm(
            logits, seq_len, beam_width=self.beam_width,
            model_path='ctc_decoder_with_lm/ctc-test-lm.binary', 
            trie_path='ctc_decoder_with_lm/ctc-test-lm.trie',
            alphabet_path='open_seq2seq/test_utils/toy_speech_data/vocab.txt',
            alpha=2.0,
            beta=0.5,
            trie_weight=0.1,
            top_paths=1, merge_repeated=False
        )
    )

    with tf.Session() as sess:
      res_greedy, res_beam, res_ixs, res_vals, res_probs = sess.run([greedy_decoded, 
          beam_search_decoded, decoded_ixs, decoded_vals, log_probabilities])

    decoded_greedy, prob_greedy = res_greedy
    decoded_text = ''.join([self.vocab[c] for c in decoded_greedy[0].values])
    self.assertTrue( abs(7079.117 + prob_greedy[0][0]) < self.tol )
    self.assertTrue( decoded_text == 'then seconds' )

    decoded_beam, prob_beam = res_beam
    decoded_text = ''.join([self.vocab[c] for c in decoded_beam[0].values])
    if tf.__version__ >= '1.11':
      # works for newer versions only (with CTC decoder fix)
      self.assertTrue( abs(1.1842575 + prob_beam[0][0]) < self.tol )
    self.assertTrue( decoded_text == 'then seconds' )

    self.assertTrue( abs(4.619581 + res_probs[0][0]) < self.tol )
    decoded_text = ''.join([self.vocab[c] for c in res_vals[0]])
    self.assertTrue( decoded_text == self.label )


  def test_beam_decoders(self):
    '''
    Test on random data that custom decoder outputs the same transcript
    if its parameters are equal to zero: alpha = beta = trie_weight = 0.0
    '''
    np.random.seed(1234)
    logits = tf.constant(np.random.uniform(size=self.seq.shape).astype(np.float32))
    seq_len = tf.constant([self.seq.shape[0]])

    beam_search_decoded = tf.nn.ctc_beam_search_decoder(logits, seq_len,
        beam_width=self.beam_width,
        top_paths=1,
        merge_repeated=False)

    custom_op_module = tf.load_op_library('ctc_decoder_with_lm/libctc_decoder_with_kenlm.so')
    decoded_ixs, decoded_vals, decoded_shapes, log_probabilities = (
        custom_op_module.ctc_beam_search_decoder_with_lm(
            logits, seq_len, beam_width=self.beam_width,
            model_path='ctc_decoder_with_lm/ctc-test-lm.binary',
            trie_path='ctc_decoder_with_lm/ctc-test-lm.trie',
            alphabet_path='open_seq2seq/test_utils/toy_speech_data/vocab.txt',
            alpha=0.0,
            beta=0.0,
            trie_weight=0.0,
            top_paths=1, merge_repeated=False
        )
    )

    with tf.Session() as sess:
      res_beam, res_ixs, res_vals, res_probs = sess.run([beam_search_decoded,
          decoded_ixs, decoded_vals, log_probabilities])

    decoded_beam, prob_beam = res_beam
    prob1 = prob_beam[0][0]
    decoded_text1 = ''.join([self.vocab[c] for c in decoded_beam[0].values])

    prob2 = res_probs[0][0]
    if tf.__version__ >= '1.11':
      # works for newer versions only (with CTC decoder fix)
      self.assertTrue( abs(prob1 - prob2) < self.tol )
    self.assertTrue( prob2 < 0 )
    decoded_text2 = ''.join([self.vocab[c] for c in res_vals[0]])

    self.assertTrue( decoded_text1 == decoded_text2 )

    
if __name__ == '__main__':
  tf.test.main()

