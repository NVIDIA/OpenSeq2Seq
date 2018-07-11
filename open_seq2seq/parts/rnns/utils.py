# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
<<<<<<< HEAD
from six.moves import range

from tensorflow.python.ops.rnn_cell import ResidualWrapper, DropoutWrapper,\
                                           MultiRNNCell
from open_seq2seq.parts.rnns.slstm import BasicSLSTMCell
from open_seq2seq.parts.rnns.glstm import GLSTMCell
from open_seq2seq.parts.rnns.zoneout import ZoneoutWrapper

def single_cell(cell_class,
                 cell_params,
                 dp_input_keep_prob=1.0,
                 dp_output_keep_prob=1.0,
                 zoneout_prob=0.,
                 training=True,
                 residual_connections=False):
  """Creates an instance of the rnn cell.
     Such cell describes one step one layer and can include residual connection
     and/or dropout

     Args:
       cell_class: Tensorflow RNN cell class
       cell_params (dict): cell parameters
       dp_input_keep_prob (float): (default: 1.0) input dropout keep
          probability.
       dp_output_keep_prob (float): (default: 1.0) output dropout keep
          probability.
       residual_connections (bool): whether to add residual connection

     Returns:
       TF RNN instance
    """
  cell = cell_class(**cell_params)
  if residual_connections:
    cell = ResidualWrapper(cell)
  if zoneout_prob > 0. and (dp_input_keep_prob < 1.0 or dp_output_keep_prob < 1.0):
    raise ValueError("Currently applying both dropout and zoneout on the same cell. This is not recommended")
  if dp_input_keep_prob != 1.0 or dp_output_keep_prob != 1.0:
    cell = DropoutWrapper(cell, input_keep_prob=dp_input_keep_prob,
                          output_keep_prob=dp_output_keep_prob)
  if zoneout_prob > 0.:
    cell = ZoneoutWrapper(cell, zoneout_prob, is_training=training)
  return cell
