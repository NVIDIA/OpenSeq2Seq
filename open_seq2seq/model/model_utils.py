# Copyright (c) 2017 NVIDIA Corporation
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import ResidualWrapper, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.rnn import GLSTMCell
from .slstm import BasicSLSTMCell

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

