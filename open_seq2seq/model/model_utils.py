# Copyright (c) 2017 NVIDIA Corporation
import os
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import ResidualWrapper, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.rnn import GLSTMCell
from .slstm import BasicSLSTMCell
from open_seq2seq.data import utils
from tensorflow.core.framework import summary_pb2

def create_rnn_cell(cell_type,
                    cell_params,
                    num_layers=1,
                    dp_input_keep_prob=1.0,
                    dp_output_keep_prob=1.0,
                    residual_connections=False,
                    wrap_to_multi_rnn=True):
  """
  TODO: MOVE THIS properly to utils. Write doc
  :param cell_type:
  :param cell_params:
  :param num_layers:
  :param dp_input_keep_prob:
  :param dp_output_keep_prob:
  :param residual_connections:
  :return:
  """

  def single_cell(cell_params):
    # TODO: This method is ugly - redo
    size = cell_params["num_units"]
    proj_size = None if "proj_size" not in cell_params else cell_params["proj_size"]

    if cell_type == "lstm":
      if not residual_connections:
        if dp_input_keep_prob == 1.0 and dp_output_keep_prob == 1.0:
          return tf.nn.rnn_cell.LSTMCell(num_units=size,
                                         num_proj=proj_size,
                                         forget_bias=1.0)
        else:
          return DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=size,
                                                        num_proj=proj_size,
                                                        forget_bias=1.0),
                                input_keep_prob=dp_input_keep_prob,
                                output_keep_prob=dp_output_keep_prob)
      else: #residual connection required
        if dp_input_keep_prob == 1.0 and dp_output_keep_prob == 1.0:
          return ResidualWrapper(tf.nn.rnn_cell.LSTMCell(num_units=size,
                                                         num_proj=proj_size,
                                                         forget_bias=1.0))
        else:
          return ResidualWrapper(DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=size,
                                                                        num_proj=proj_size,
                                                                        forget_bias=1.0),
                                 input_keep_prob=dp_input_keep_prob,
                                 output_keep_prob=dp_output_keep_prob))
    elif cell_type == "gru":
      if not residual_connections:
        if dp_input_keep_prob == 1.0 and dp_output_keep_prob == 1.0:
          return tf.nn.rnn_cell.GRUCell(num_units=size)
        else:
          return DropoutWrapper(tf.nn.rnn_cell.GRUCell(num_units=size),
                                input_keep_prob=dp_input_keep_prob,
                                output_keep_prob=dp_output_keep_prob
                                )
      else: #residual connection required
        if dp_input_keep_prob == 1.0 and dp_output_keep_prob == 1.0:
          return ResidualWrapper(tf.nn.rnn_cell.GRUCell(num_units=size))
        else:
          return ResidualWrapper(DropoutWrapper(tf.nn.rnn_cell.GRUCell(num_units=size),
                                                input_keep_prob=dp_input_keep_prob,
                                                output_keep_prob=dp_output_keep_prob))
    elif cell_type == "glstm":
      num_groups = cell_params["num_groups"]
      if not residual_connections:
        if dp_input_keep_prob == 1.0 and dp_output_keep_prob == 1.0:
          return GLSTMCell(num_units=size,
                           number_of_groups=num_groups,
                           num_proj=proj_size,
                           forget_bias=1.0)
        else:
          return DropoutWrapper(GLSTMCell(num_units=size,
                                          number_of_groups=num_groups,
                                          num_proj=proj_size,
                                          forget_bias=1.0),
                                input_keep_prob=dp_input_keep_prob,
                                output_keep_prob=dp_output_keep_prob)
      else:  # residual connection required
        if dp_input_keep_prob == 1.0 and dp_output_keep_prob == 1.0:
          return ResidualWrapper(GLSTMCell(num_units=size,
                                           number_of_groups=num_groups,
                                           num_proj=proj_size,
                                           forget_bias=1.0))
        else:
          return ResidualWrapper(DropoutWrapper(GLSTMCell(num_units=size,
                                                          number_of_groups=num_groups,
                                                          num_proj=proj_size,
                                                          forget_bias=1.0),
                                                input_keep_prob=dp_input_keep_prob,
                                                output_keep_prob=dp_output_keep_prob))
    elif cell_type == "slstm":
      if not residual_connections:
        if dp_input_keep_prob == 1.0 and dp_output_keep_prob == 1.0:
          return BasicSLSTMCell(num_units=size)
        else:
          return DropoutWrapper(BasicSLSTMCell(num_units=size),
                                input_keep_prob=dp_input_keep_prob,
                                output_keep_prob=dp_output_keep_prob
                                )
      else: #residual connection required
        if dp_input_keep_prob == 1.0 and dp_output_keep_prob == 1.0:
          return ResidualWrapper(BasicSLSTMCell(num_units=size))
        else:
          return ResidualWrapper(DropoutWrapper(BasicSLSTMCell(num_units=size),
                                                input_keep_prob=dp_input_keep_prob,
                                                output_keep_prob=dp_output_keep_prob))
    else:
      raise ValueError("Unknown RNN cell class: {}".format(cell_type))

  if num_layers > 1:
    if wrap_to_multi_rnn:
      return MultiRNNCell([single_cell(cell_params) for _ in range(num_layers)])
    else:
      cells = [] # for GNMT-like attention in decoder
      for i in range(num_layers):
        cells.append(single_cell(cell_params))
      return cells
  else:
    return single_cell(cell_params)


def getdtype():
  return tf.float32

def deco_print(line):
  print(">==================> " + line)


class SaveAtEnd(tf.train.SessionRunHook):
  """Session Hook which forces model save at the end of the session
  """
  def __init__(self, logdir, global_step):
    super(tf.train.SessionRunHook, self).__init__()
    self._logdir = logdir
    self._global_step = global_step

  def begin(self):
    self._saver = tf.train.Saver()

  def end(self, session):
    deco_print("Saving last checkpoint")
    self._saver.save(session, save_path=os.path.join(self._logdir, "model"), global_step=self._global_step)


class EvalHook(tf.train.SessionRunHook):
  """Session Hook which performs evaluation
  """
  def __init__(self, evaluation_model,
               eval_dl,
               global_step,
               eval_frequency,
               summary_writer,
               eval_use_beam_search = True,
               eval_using_bleu = True,
               bpe_used = True,
               delimiter = " "):
    self._e_model = evaluation_model
    self._eval_dl = eval_dl
    self._global_step = global_step
    self._eval_frequency = eval_frequency
    self._eval_fetches = [self._e_model.final_outputs]
    self._eval_use_beam_search = eval_use_beam_search
    self._eval_using_bleu = eval_using_bleu
    self._bpe_used = bpe_used
    self._sw = summary_writer
    self._delimiter = delimiter

  def after_run(self, run_context, run_values):
    sess = run_context.session
    i = sess.run(self._global_step)
    if i % self._eval_frequency == 0 and i > 0:
      deco_print("Evaluation on validation set")
      preds = []
      targets = []
      # iterate through evaluation data
      for j, (ex, ey, ebucket_id, elen_x, elen_y) in enumerate(self._eval_dl.iterate_one_epoch()):
        samples = sess.run(fetches = self._eval_fetches,
                           feed_dict={
                             self._e_model.x: ex,
                             self._e_model.x_length: elen_x,
                             self._e_model.y: ey,
                             self._e_model.y_length: elen_y
                           })
        samples = samples[0].predicted_ids[:, :, 0] if self._eval_use_beam_search else samples[0].sample_id

        if self._eval_using_bleu:
          preds.extend([utils.transform_for_bleu(si,
                                                 vocab=self._eval_dl.target_idx2seq,
                                                 ignore_special=True,
                                                 delim=self._delimiter, bpe_used=self._bpe_used) for sample in
                        [samples] for si in sample])
          targets.extend([[utils.transform_for_bleu(yi,
                                                    vocab=self._eval_dl.target_idx2seq,
                                                    ignore_special=True,
                                                    delim=self._delimiter, bpe_used=self._bpe_used)] for yii in [ey]
                          for yi in yii])

      self._eval_dl.bucketize()

      if self._eval_using_bleu:
        eval_bleu = utils.calculate_bleu(preds, targets)
        bleu_value = summary_pb2.Summary.Value(tag="Eval_BLEU_Score", simple_value=eval_bleu)
        bleu_summary = summary_pb2.Summary(value=[bleu_value])
        self._sw.add_summary(summary=bleu_summary, global_step=sess.run(self._global_step))
        self._sw.flush()
