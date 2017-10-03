import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.cudnn_rnn.python.ops.cudnn_rnn_ops import CudnnLSTM
from tensorflow.python.ops import variable_scope as vs

class cudnn_LSTMCell(rnn_cell_impl.RNNCell):
	def __init__(self, num_units, input_size, direction='unidirectional',
				use_peepholes=False, cell_clip=None,
				initializer=None, num_proj=None, proj_clip=None,
				num_unit_shards=None, num_proj_shards=None,
				forget_bias=0.0, state_is_tuple=True,
				activation=None, reuse=None):
		super(cudnn_LSTMCell, self).__init__(_reuse=reuse)
		if not state_is_tuple:
			logging.warn("%s: Using a concatenated state is slower and will soon be "
						"deprecated.  Use state_is_tuple=True.", self)
		if num_unit_shards is not None or num_proj_shards is not None:
			logging.warn(
				"%s: The num_unit_shards and proj_unit_shards parameters are "
				"deprecated and will be removed in Jan 2017.  "
				"Use a variable scope with a partitioner instead.", self)

		### Error Message
		if use_peepholes:
			raise ValueError("Using peepholes is not supported. ")
		###

		self._num_units = num_units
		self._input_size = input_size
		self._direction = direction
		self._use_peepholes = use_peepholes
		self._cell_clip = cell_clip
		self._initializer = initializer
		self._num_proj = num_proj
		self._proj_clip = proj_clip
		self._num_unit_shards = num_unit_shards
		self._num_proj_shards = num_proj_shards
		self._forget_bias = forget_bias
		self._state_is_tuple = state_is_tuple
		self._activation = activation or math_ops.tanh

		self._CudnnLSTM = CudnnLSTM(num_layers=1, 
									num_units=num_units,
									input_size=input_size,
									direction=direction)
		self._params_size = LSTM_params_size(1, num_units, input_size)
		if direction == 'bidirectional':
			self._params_size *= 2
		self._params = vs.get_variable(
			"params", shape=[self._params_size], dtype=tf.float32)

		if num_proj:
			self._state_size = (
				rnn_cell_impl.LSTMStateTuple(num_units, num_proj)
				if state_is_tuple else num_units + num_proj)
			self._output_size = num_proj
		else:
			self._state_size = (
				rnn_cell_impl.LSTMStateTuple(num_units, num_units)
				if state_is_tuple else 2 * num_units)
			self._output_size = num_units

	@property
	def state_size(self):
		return self._state_size

	@property
	def output_size(self):
		return self._output_size

	def call(self, inputs, state):		
		num_proj = self._num_units if self._num_proj is None else self._num_proj

		if self._state_is_tuple:
			(c_prev, m_prev) = state
		else:
			c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
			m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

		scope = vs.get_variable_scope()
		with vs.variable_scope(scope, initializer=self._initializer) as unit_scope:
			if self._num_unit_shards is not None:
				unit_scope.set_partitioner(
					partitioned_variables.fixed_size_partitioner(
						self._num_unit_shards))

			_, output_h, output_c = self._CudnnLSTM(
				input_data=array_ops.expand_dims(inputs, [0]),
				input_h=array_ops.expand_dims(m_prev, [0]),
				input_c=array_ops.expand_dims(c_prev, [0]),
				params=self._params)

			c = array_ops.squeeze(output_c, [0])
			m = array_ops.squeeze(output_h, [0])

			if self._cell_clip is not None:
				# pylint: disable=invalid-unary-operand-type
				c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
				# pylint: enable=invalid-unary-operand-type

			if self._num_proj is not None:
				with vs.variable_scope("projection") as proj_scope:
					if self._num_proj_shards is not None:
						proj_scope.set_partitioner(
							partitioned_variables.fixed_size_partitioner(
								self._num_proj_shards))
					m = rnn_cell_impl._linear(m, self._num_proj, bias=False)

				if self._proj_clip is not None:
					# pylint: disable=invalid-unary-operand-type
					m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
					# pylint: enable=invalid-unary-operand-type

			new_state = (rnn_cell_impl.LSTMStateTuple(c, m) if self._state_is_tuple else
						array_ops.concat([c, m], 1))
		return m, new_state

def LSTM_params_size(num_layers, num_units, input_size):
	params_size = (input_size * num_units * 4) + (num_units * num_units * 4) + (num_units * 2 * 4)
	for _ in range(num_layers - 1):
		next_input_size = num_units * 2
		params_size += (next_input_size * num_units * 4) + (num_units * num_units * 4) + (num_units * 2 * 4)
	return params_size
