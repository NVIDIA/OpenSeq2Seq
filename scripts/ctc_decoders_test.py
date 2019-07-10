import numpy as np
import pickle
import tensorflow as tf

from ctc_decoders import Scorer, ctc_beam_search_decoder



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

def softmax(x):
  m = np.expand_dims(np.max(x, axis=-1), -1)
  e = np.exp(x - m)
  return e / np.expand_dims(e.sum(axis=-1), -1)


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

    with tf.Session() as sess:
      res_greedy, res_beam = sess.run([greedy_decoded, 
          beam_search_decoded])

    decoded_greedy, prob_greedy = res_greedy
    decoded_text = ''.join([self.vocab[c] for c in decoded_greedy[0].values])
    self.assertTrue( abs(7079.117 + prob_greedy[0][0]) < self.tol )
    self.assertTrue( decoded_text == 'then seconds' )

    decoded_beam, prob_beam = res_beam
    decoded_text = ''.join([self.vocab[c] for c in decoded_beam[0].values])
    if tf.__version__ >= '1.11':
      # works for newer versions only (with CTC decoder fix)
      self.assertTrue( abs(1.1842 + prob_beam[0][0]) < self.tol )
    self.assertTrue( decoded_text == 'then seconds' )

    scorer = Scorer(alpha=2.0, beta=0.5,
        model_path='ctc_decoder_with_lm/ctc-test-lm.binary', 
        vocabulary=self.vocab[:-1])
    res = ctc_beam_search_decoder(softmax(self.seq.squeeze()), self.vocab[:-1],
                                  beam_size=self.beam_width,
                                  ext_scoring_func=scorer)
    res_prob, decoded_text = res[0]
    self.assertTrue( abs(4.0845 + res_prob) < self.tol )
    self.assertTrue( decoded_text == self.label )


  def test_beam_decoders(self):
    '''
    Test on random data that custom decoder outputs the same transcript
    as standard TF beam search decoder
    '''
    seq = np.random.uniform(size=self.seq.shape).astype(np.float32)
    logits = tf.constant(seq)
    seq_len = tf.constant([self.seq.shape[0]])

    beam_search_decoded = tf.nn.ctc_beam_search_decoder(logits, seq_len,
        beam_width=self.beam_width,
        top_paths=1,
        merge_repeated=False)


    with tf.Session() as sess:
      res_beam = sess.run(beam_search_decoded)
    decoded_beam, prob_beam = res_beam
    prob1 = prob_beam[0][0]
    decoded_text1 = ''.join([self.vocab[c] for c in decoded_beam[0].values])

    res = ctc_beam_search_decoder(softmax(seq.squeeze()), self.vocab[:-1],
                                  beam_size=self.beam_width)
    prob2, decoded_text2 = res[0]

    if tf.__version__ >= '1.11':
      # works for newer versions only (with CTC decoder fix)
      self.assertTrue( abs(prob1 - prob2) < self.tol )
    self.assertTrue( prob2 < 0 )

    self.assertTrue( decoded_text1 == decoded_text2 )

    
if __name__ == '__main__':
  tf.test.main()

