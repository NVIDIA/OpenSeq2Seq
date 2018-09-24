import random
import numpy as np
import tensorflow as tf

from .encoder_decoder import EncoderDecoderModel
from open_seq2seq.data import WKTDataLayer
from open_seq2seq.utils.utils import deco_print, array_to_string
from open_seq2seq.utils import metrics

class LSTMLM(EncoderDecoderModel):
  """
  An example class implementing an LSTM language model.
  """
  def __init__(self, params, mode="train", hvd=None):
    super(EncoderDecoderModel, self).__init__(params=params, mode=mode, hvd=hvd)

    if 'encoder_params' not in self.params:
      self.params['encoder_params'] = {}
    if 'decoder_params' not in self.params:
      self.params['decoder_params'] = {}
    if 'loss_params' not in self.params:
      self.params['loss_params'] = {}

    self._lm_phase = isinstance(self.get_data_layer(), WKTDataLayer)

    self._encoder = self._create_encoder()
    self._decoder = self._create_decoder()
    if self.mode == 'train' or self.mode == 'eval':
      self._loss_computator = self._create_loss()
    else:
      self._loss_computator = None

    self.delimiter = self.get_data_layer().delimiter

  def _create_encoder(self):
    self._print_f1 = False
    self.params['encoder_params']['vocab_size'] = (
      self.get_data_layer().vocab_size
    )
    self.params['encoder_params']['end_token'] = (
      self.get_data_layer().end_token
    )
    self.params['encoder_params']['batch_size'] = (
      self.get_data_layer().batch_size
    )
    if not self._lm_phase:
      self.params['encoder_params']['fc_dim'] = (
        self.get_data_layer().num_classes
      )
      if self.params['encoder_params']['fc_dim'] == 2:
        self._print_f1 = True
    if self._lm_phase:
      self.params['encoder_params']['seed_tokens'] = (
        self.get_data_layer().params['seed_tokens']
      )
    return super(LSTMLM, self)._create_encoder()

  def _create_loss(self):
    if self._lm_phase:
      self.params['loss_params']['batch_size'] = (
        self.get_data_layer().batch_size
      )
      self.params['loss_params']['tgt_vocab_size'] = (
        self.get_data_layer().vocab_size
      )

    return super(LSTMLM, self)._create_loss()


  def infer(self, input_values, output_values):
    if self._lm_phase:
      vocab = self.get_data_layer().corp.dictionary.idx2word
      seed_tokens = self.params['encoder_params']['seed_tokens']
      for i in range(len(seed_tokens)):
        print('Seed:', vocab[seed_tokens[i]] + '\n')
        deco_print(
          "Output: " + array_to_string(
            output_values[0][i],
            vocab=self.get_data_layer().corp.dictionary.idx2word,
            delim=self.delimiter,
          ),
          offset=4,
        )
      return []
    else:
      ex, elen_x = input_values['source_tensors']
      ey, elen_y = None, None
      if 'target_tensors' in input_values:
        ey, elen_y = input_values['target_tensors']

      n_samples = len(ex)
      results = []
      for i in range(n_samples):
        current_x = array_to_string(
          ex[i][:elen_x[i]],
          vocab=self.get_data_layer().corp.dictionary.idx2word,
          delim=self.delimiter,
        ),
        current_pred = np.argmax(output_values[0][i])
        curret_y = None
        if ey is not None:
          current_y = np.argmax(ey[i])

        results.append((current_x[0], current_pred, current_y))
      return results
  

  def maybe_print_logs(self, input_values, output_values, training_step):
    x, len_x = input_values['source_tensors']
    y, len_y = input_values['target_tensors']    

    x_sample = x[0]
    len_x_sample = len_x[0]
    y_sample = y[0]
    len_y_sample = len_y[0]

    deco_print(
      "Train Source[0]:     " + array_to_string(
        x_sample[:len_x_sample],
        vocab=self.get_data_layer().corp.dictionary.idx2word,
        delim=self.delimiter,
      ),
      offset=4,
    )

    if self._lm_phase:
      deco_print(
        "Train Target[0]:     " + array_to_string(
          y_sample[:len_y_sample],
          vocab=self.get_data_layer().corp.dictionary.idx2word,
          delim=self.delimiter,
        ),
        offset=4,
      )
    else:
      deco_print(
        "TRAIN Target[0]:     " + str(np.argmax(y_sample)),
        offset=4,
      )
      samples = output_values[0][0]
      deco_print(
        "TRAIN Prediction[0]:     " + str(samples),
        offset=4,
      )
      labels = np.argmax(y, 1)
      preds = np.argmax(output_values[0], axis=-1)
      print('Labels', labels)
      print('Preds', preds)

      deco_print(
        "Accuracy: {:.4f}".format(metrics.accuracy(labels, preds)),
        offset = 4,
      )

      if self._print_f1:
        deco_print(
          "Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}"
              .format(metrics.precision(labels, preds), 
                      metrics.recall(labels, preds),
                      metrics.f1(labels, preds)),
          offset = 4,
        )

    return {}

  def evaluate(self, input_values, output_values):
    ex, elen_x = input_values['source_tensors']
    ey, elen_y = input_values['target_tensors']

    x_sample = ex[0]
    len_x_sample = elen_x[0]
    y_sample = ey[0]
    len_y_sample = elen_y[0]
    
    return_values = {}
    
    if self._lm_phase:
      flip = random.random()
      if flip <= 0.9:
        return return_values

      deco_print(
        "*****EVAL Source[0]:     " + array_to_string(
          x_sample[:len_x_sample],
          vocab=self.get_data_layer().corp.dictionary.idx2word,
          delim=self.delimiter,
        ),
        offset=4,
      )
      samples = np.argmax(output_values[0][0], axis=-1)
      deco_print(
        "*****EVAL Target[0]:     " + array_to_string(
          y_sample[:len_y_sample],
          vocab=self.get_data_layer().corp.dictionary.idx2word,
          delim=self.delimiter,
        ),
        offset=4,
      )
    
      deco_print(
        "*****EVAL Prediction[0]: " + array_to_string(
          samples,
          vocab=self.get_data_layer().corp.dictionary.idx2word,
          delim=self.delimiter,
        ),
        offset=4,
      )
    else:
      deco_print(
        "*****EVAL Source[0]:     " + array_to_string(
          x_sample[:len_x_sample],
          vocab=self.get_data_layer().corp.dictionary.idx2word,
          delim=self.delimiter,
        ),
        offset=4,
      )
      samples = output_values[0][0]
      deco_print(
        "EVAL Target[0]:     " + str(np.argmax(y_sample)),
        offset=4,
      )
      deco_print(
        "EVAL Prediction[0]:     " + str(samples),
        offset=4,
      )

      labels = np.argmax(ey, 1)
      preds = np.argmax(output_values[0], axis=-1)
      print('Labels', labels)
      print('Preds', preds)

      return_values['accuracy'] = metrics.accuracy(labels, preds)

      if self._print_f1:
        return_values['true_pos'] = metrics.true_positives(labels, preds)
        return_values['pred_pos'] = np.sum(preds)
        return_values['actual_pos'] = np.sum(labels)

    return return_values

  def finalize_evaluation(self, results_per_batch, training_step=None):
    accuracies = []
    true_pos, pred_pos, actual_pos = 0.0, 0.0, 0.0

    for results in results_per_batch:
      if not 'accuracy' in results:
        return {}
      accuracies.append(results['accuracy'])
      if 'true_pos' in results:
        true_pos += results['true_pos']
        pred_pos += results['pred_pos']
        actual_pos += results['actual_pos']

    deco_print(
      "EVAL Accuracy: {:.4f}".format(np.mean(accuracies)),
      offset = 4,
    )

    if true_pos > 0:
      prec = true_pos / pred_pos
      rec = true_pos / actual_pos
      f1 = 2.0 * prec * rec / (rec + prec)
      deco_print(
        "EVAL Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f} | True pos: {}"
            .format(prec, rec, f1, true_pos),
        offset = 4,
      )
    return {}

  def finalize_inference(self, results_per_batch, output_file):
    out = open(output_file, 'w')
    out.write('\t'.join(['Source', 'Pred', 'Label']) + '\n')
    preds, labels = [], []

    for results in results_per_batch:
      for x, pred, y in results:
        out.write('\t'.join([x, str(pred), str(y)]) + '\n')
        preds.append(pred)
        labels.append(y)

    if len(labels) > 0 and labels[0] is not None:
      preds = np.asarray(preds)
      labels = np.asarray(labels)
      deco_print(
        "TEST Accuracy: {:.4f}".format(metrics.accuracy(labels, preds)),
        offset = 4,
      )
      deco_print(
        "TEST Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}"
            .format(metrics.precision(labels, preds), 
                    metrics.recall(labels, preds),
                    metrics.f1(labels, preds)),
        offset = 4,
      )
    return {}

  def _get_num_objects_per_step(self, worker_id=0):
    """Returns number of source tokens + number of target tokens in batch."""
    data_layer = self.get_data_layer(worker_id)
    # sum of source length in batch
    num_tokens = tf.reduce_sum(data_layer.input_tensors['source_tensors'][1])
    if self.mode != "infer":
      # sum of target length in batch
      num_tokens += tf.reduce_sum(data_layer.input_tensors['target_tensors'][1])
    else:
      # TODO: this is not going to be correct when batch size > 1, since it will
      #       count padding?
      num_tokens += tf.reduce_sum(tf.shape(self.get_output_tensors(worker_id)[0]))
    return num_tokens
