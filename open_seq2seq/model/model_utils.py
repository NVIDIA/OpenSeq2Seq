# Copyright (c) 2017 NVIDIA Corporation
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell, ResidualWrapper, DropoutWrapper, MultiRNNCell
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
    if cell_type == "lstm":
      cell_class = LSTMCell
    elif cell_type == "gru":
      cell_class = GRUCell
    elif cell_type == "glstm":
      cell_class = GLSTMCell
      num_groups = 4
    elif cell_type == "slstm":
      cell_class = BasicSLSTMCell
    else:
      raise ValueError("Unknown RNN cell class: {}".format(cell_type))

    if residual_connections:
      if dp_input_keep_prob !=1.0 or dp_output_keep_prob != 1.0:
        if cell_type != "glstm":
          return DropoutWrapper(ResidualWrapper(cell_class(num_units=size)),
                              input_keep_prob=dp_input_keep_prob,
                              output_keep_prob=dp_output_keep_prob)
        else:
          return DropoutWrapper(ResidualWrapper(cell_class(num_units=size, number_of_groups=num_groups)),
                                input_keep_prob=dp_input_keep_prob,
                                output_keep_prob=dp_output_keep_prob)
      else:
        if cell_type != "glstm":
          return ResidualWrapper(cell_class(num_units=size))
        else:
          return ResidualWrapper(cell_class(num_units=size, number_of_groups=num_groups))
    else:
      if dp_input_keep_prob !=1.0 or dp_output_keep_prob != 1.0:
        if cell_type != "glstm":
          return DropoutWrapper(cell_class(num_units=size),
                              input_keep_prob=dp_input_keep_prob,
                              output_keep_prob=dp_output_keep_prob)
        else:
          return DropoutWrapper(cell_class(num_units=size, number_of_groups=num_groups),
                                input_keep_prob=dp_input_keep_prob,
                                output_keep_prob=dp_output_keep_prob)
      else:
        if cell_type != "glstm":
          return cell_class(num_units=size)
        else:
          return cell_class(num_units=size, number_of_groups=num_groups)

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

