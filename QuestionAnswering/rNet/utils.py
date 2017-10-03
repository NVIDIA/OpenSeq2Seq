import random
import tensorflow as tf
from .cudnn_rnn_cell_impl import cudnn_LSTMCell
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell, ResidualWrapper, DropoutWrapper, MultiRNNCell

def deco_print(line):
	print(">==================> " + line)

def getdtype():
	return tf.float32

def create_rnn_cell(cell_type,
					num_units,
					num_layers=1,
					dp_input_keep_prob=1.0,
					dp_output_keep_prob=1.0,
					residual_connections=False):
	"""
	TODO: MOVE THIS properly to utils. Write doc
	:param cell_type:
	:param num_units:
	:param num_layers:
	:param dp_input_keep_prob:
	:param dp_output_keep_prob:
	:param residual_connections:
	:return:
	"""

	def single_cell(num_units):
		if cell_type == "lstm":
			cell_class = LSTMCell
		elif cell_type == "gru":
			cell_class = GRUCell

		if residual_connections:
			if dp_input_keep_prob !=1.0 or dp_output_keep_prob != 1.0:
				return DropoutWrapper(ResidualWrapper(cell_class(num_units=num_units)),
									input_keep_prob=dp_input_keep_prob,
									output_keep_prob=dp_output_keep_prob)
			else:
				return ResidualWrapper(cell_class(num_units=num_units))
		else:
			if dp_input_keep_prob !=1.0 or dp_output_keep_prob != 1.0:
				return DropoutWrapper(cell_class(num_units=num_units),
									input_keep_prob=dp_input_keep_prob,
									output_keep_prob=dp_output_keep_prob)
			else:
				return cell_class(num_units=num_units)

	if num_layers > 1:
		return MultiRNNCell([single_cell(num_units) for _ in range(num_layers)])
	else:
		return single_cell(num_units)

def create_cudnn_LSTM_cell(num_units,
						input_size,
						num_layers=1,
						dp_input_keep_prob=1.0,
						dp_output_keep_prob=1.0):
	def single_cell(name):
		with tf.variable_scope(name):
			if dp_input_keep_prob != 1.0 or dp_output_keep_prob !=1.0:
				return DropoutWrapper(cudnn_LSTMCell(num_units=num_units, input_size=input_size, direction='unidirectional'),
									input_keep_prob=dp_input_keep_prob,
									output_keep_prob=dp_output_keep_prob)
			else:
				return cudnn_LSTMCell(num_units=num_units, input_size=input_size, direction='unidirectional')

	if num_layers > 1:
		return MultiRNNCell([single_cell('layer_%d'%i) for i in range(num_layers)])
	else:
		return single_cell('layer_0')

def LSTM_params_size(num_layers, num_units, input_size):
	params_size = ((input_size * num_units * 4) + (num_units * num_units * 4) + (num_units * 2 * 4)) * 2
	for _ in range(num_layers - 1):
		next_input_size = num_units * 2
		params_size += ((next_input_size * num_units * 4) + (num_units * num_units * 4) + (num_units * 2 * 4)) * 2
	return params_size

def GRU_params_size(num_layers, num_units, input_size):
	params_size = ((input_size * num_units * 3) + (num_units * num_units * 3) + (num_units *2 * 3)) * 2
	for _ in range(num_layers - 1):
		next_input_size = num_units * 2
		params_size += ((next_input_size * num_units * 3) + (num_units * num_units * 3) + (num_units * 2 * 3)) * 2
	return params_size