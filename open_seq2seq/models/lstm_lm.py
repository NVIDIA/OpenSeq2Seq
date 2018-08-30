import tensorflow as tf

from .encoder_decoder import EncoderDecoderModel
from open_seq2seq.utils.utils import deco_print, array_to_string

class LSTMLM(EncoderDecoderModel):
  """
  An example class implementing classical text-to-text model.
  """

  def _create_encoder(self):
    self.params['encoder_params']['vocab_size'] = (
      self.get_data_layer().params['vocab_size']
    )
    self.params['encoder_params']['output_dim'] = (
      self.get_data_layer().params['vocab_size']
    )
    self.params['encoder_params']['end_token'] = (
      self.get_data_layer().params['end_token']
    )
    self.params['encoder_params']['seed_tokens'] = (
      self.get_data_layer().params['seed_tokens']
    )
    self.params['encoder_params']['batch_size'] = (
      self.get_data_layer().params['batch_size']
    )
    return super(LSTMLM, self)._create_encoder()

  def _create_loss(self):
    # self.params['loss_params']['batch_size'] = self.params['batch_size_per_gpu']
    self.params['loss_params']['batch_size'] = self.get_data_layer().params['batch_size']
    
    self.params['loss_params']['tgt_vocab_size'] = (
      self.get_data_layer().params['vocab_size']
    )

    return super(LSTMLM, self)._create_loss()


  def infer(self, input_values, output_values):
    vocab = self.get_data_layer().corp.dictionary.idx2word
    seed_tokens = self.params['encoder_params']['seed_tokens']
    for i in range(len(seed_tokens)):
      print(output_values[0][i].shape)
      print('Seed:', vocab[seed_tokens[i]] + '\n')
      deco_print(
        "Output: " + array_to_string(
          output_values[0][i],
          vocab=self.get_data_layer().corp.dictionary.idx2word,
          delim=self.get_data_layer().params["delimiter"],
        ),
        offset=4,
      )

  def maybe_print_logs(self, input_values, output_values, training_step):
    x, len_x = input_values['source_tensors']
    y, len_y = input_values['target_tensors']
    # samples = output_values[0][0]

    x_sample = x[0]
    len_x_sample = len_x[0]
    y_sample = y[0]
    len_y_sample = len_y[0]

    deco_print(
      "Train Source[0]:     " + array_to_string(
        x_sample[:len_x_sample],
        vocab=self.get_data_layer().corp.dictionary.idx2word,
        delim=self.get_data_layer().params["delimiter"],
      ),
      offset=4,
    )
    deco_print(
      "Train Target[0]:     " + array_to_string(
        y_sample[:len_y_sample],
        vocab=self.get_data_layer().corp.dictionary.idx2word,
        delim=self.get_data_layer().params["delimiter"],
      ),
      offset=4,
    )

    return {}

  def evaluate(self, input_values, output_values):
    ex, elen_x = input_values['source_tensors']
    ey, elen_y = input_values['target_tensors']

    x_sample = ex[0]
    len_x_sample = elen_x[0]
    y_sample = ey[0]
    len_y_sample = elen_y[0]

    deco_print(
      "*****EVAL Source[0]:     " + array_to_string(
        x_sample[:len_x_sample],
        vocab=self.get_data_layer().corp.dictionary.idx2word,
        delim=self.get_data_layer().params["delimiter"],
      ),
      offset=4,
    )
    deco_print(
      "*****EVAL Target[0]:     " + array_to_string(
        y_sample[:len_y_sample],
        vocab=self.get_data_layer().corp.dictionary.idx2word,
        delim=self.get_data_layer().params["delimiter"],
      ),
      offset=4,
    )
    samples = output_values[0][0]
    deco_print(
      "*****EVAL Prediction[0]: " + array_to_string(
        samples,
        vocab=self.get_data_layer().corp.dictionary.idx2word,
        delim=self.get_data_layer().params["delimiter"],
      ),
      offset=4,
    )

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
