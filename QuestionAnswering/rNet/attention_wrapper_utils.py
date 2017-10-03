import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import *

_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access

######
# Remove later
######
class GatedAttentionWrapper(rnn_cell_impl.RNNCell):
	""" Wraps 'RNNCell' with gated-attention
		Modifies https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
	"""
	def __init__(self,
				cell,
				attention_mechanism,
				attention_layer_size=None,
				alignment_history=False,
				cell_input_fn=None,
				output_attention=True,
				initial_cell_state=None,
				name=None):
		super(GatedAttentionWrapper, self).__init__(name=name)
		if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
			raise TypeError("cell must be an RNNCell, saw type: %s" % type(cell).__name__)
		if not isinstance(attention_mechanism, AttentionMechanism):
			raise TypeError("attention_mechanism must be a AttentionMechanism, saw type: %s"
				% type(attention_mechanism).__name__)
		if cell_input_fn is None:
			cell_input_fn = (lambda inputs, attention: array_ops.concat([inputs, attention], -1))
		else:
			if not callable(cell_input_fn):
				raise TypeError("cell_input_fn must be callable, saw type: %s"
					% type(cell_input_fn).__name__)

		if attention_layer_size is not None:
			self._attention_layer = layers_core.Dense(attention_layer_size, name="attention_layer", use_bias=False)
			self._attention_layer_size = attention_layer_size
		else:
			self._attention_layer = None
			self._attention_layer_size = attention_mechanism.values.get_shape()[-1].value

		self._cell = cell
		self._attention_mechanism = attention_mechanism
		self._cell_input_fn = cell_input_fn
		self._output_attention = output_attention
		self._alignment_history = alignment_history
		with ops.name_scope(name, "AttentionWrapperInit"):
			if initial_cell_state is None:
				self._initial_cell_state = None
			else:
				final_state_tensor = nest.flatten(initial_cell_state)[-1]
				state_batch_size = (
					final_state_tensor.shape[0].value
					or array_ops.shape(final_state_tensor)[0])
				error_message = (
					"When constructing AttentionWrapper %s: " % self._base_name +
					"Non-matching batch sizes between the memory "
					"(encoder output) and initial_cell_state.  Are you using "
					"the BeamSearchDecoder?  You may need to tile your initial state "
					"via the tf.contrib.seq2seq.tile_batch function with argument "
					"multiple=beam_width.")
				with ops.control_dependencies(
					[check_ops.assert_equal(state_batch_size,
											self._attention_mechanism.batch_size,
											message=error_message)]):
					self._initial_cell_state = nest.map_structure(
						lambda s: array_ops.identity(s, name="check_initial_cell_state"),
						initial_cell_state)

	@property
	def output_size(self):
		if self._output_attention:
			return self._attention_layer_size
		else:
			return self._cell.output_size

	@property
	def state_size(self):
		return AttentionWrapperState(
			cell_state=self._cell.state_size,
			time=tensor_shape.TensorShape([]),
			attention=self._attention_layer_size,
			alignments=self._attention_mechanism.alignments_size,
			alignment_history=())  # alignment_history is sometimes a TensorArray

	def zero_state(self, batch_size, dtype):
		with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
			if self._initial_cell_state is not None:
				cell_state = self._initial_cell_state
			else:
				cell_state = self._cell.zero_state(batch_size, dtype)
			error_message = (
				"When calling zero_state of AttentionWrapper %s: " % self._base_name +
				"Non-matching batch sizes between the memory "
				"(encoder output) and the requested batch size.  Are you using "
				"the BeamSearchDecoder?  If so, make sure your encoder output has "
				"been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
				"the batch_size= argument passed to zero_state is "
				"batch_size * beam_width.")
			with ops.control_dependencies(
				[check_ops.assert_equal(batch_size,
										self._attention_mechanism.batch_size,
										message=error_message)]):
				cell_state = nest.map_structure(
					lambda s: array_ops.identity(s, name="checked_cell_state"),
					cell_state)
			if self._alignment_history:
				alignment_history = tensor_array_ops.TensorArray(
					dtype=dtype, size=0, dynamic_size=True)
			else:
				alignment_history = ()
			return AttentionWrapperState(
				cell_state=cell_state,
				time=array_ops.zeros([], dtype=dtypes.int32),
				attention=_zero_state_tensors(self._attention_layer_size, batch_size, dtype),
				alignments=self._attention_mechanism.initial_alignments(batch_size, dtype),
				alignment_history=alignment_history)

	def call(self, inputs, state):
		cell_inputs = self._cell_input_fn(inputs, state.attention)
		cell_state = state.cell_state
		cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

		cell_batch_size = (
			cell_output.shape[0].value or array_ops.shape(cell_output)[0])
		error_message = (
			"When applying AttentionWrapper %s: " % self.name +
			"Non-matching batch sizes between the memory "
			"(encoder output) and the query (decoder output).  Are you using "
			"the BeamSearchDecoder?  You may need to tile your memory input via "
			"the tf.contrib.seq2seq.tile_batch function with argument "
			"multiple=beam_width.")
		with ops.control_dependencies(
			[check_ops.assert_equal(cell_batch_size,
									self._attention_mechanism.batch_size,
									message=error_message)]):
			cell_output = array_ops.identity(
				cell_output, name="checked_cell_output")

		alignments = self._attention_mechanism(
			array_ops.concat([cell_output, inputs], 1), previous_alignments=state.alignments) # change

		expanded_alignments = array_ops.expand_dims(alignments, 1)
		attention_mechanism_values = self._attention_mechanism.values
		context = math_ops.matmul(expanded_alignments, attention_mechanism_values)
		context = array_ops.squeeze(context, [1])

		if self._attention_layer is not None:
			attention = self._attention_layer(
				array_ops.concat([cell_output, context], 1))
		else:
			attention = context

		if self._alignment_history:
			alignment_history = state.alignment_history.write(
				state.time, alignments)
		else:
			alignment_history = ()

		next_state = AttentionWrapperState(
			time=state.time + 1,
			cell_state=next_cell_state,
			attention=attention,
			alignments=alignments,
			alignment_history=alignment_history)

		if self._output_attention:
			return attention, next_state
		else:
			return cell_output, next_state

class AttentionWrapper_UserDefined(rnn_cell_impl.RNNCell):
	""" Wraps 'RNNCell' with gated-attention
		Modifies https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
	"""
	def __init__(self,
				cell,
				attention_mechanism,
				attention_layer_size=None,
				alignment_history=False,
				cell_input_fn=None,
				output_attention=True,
				initial_cell_state=None,
				name=None):
		super(AttentionWrapper_UserDefined, self).__init__(name=name)
		if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
			raise TypeError("cell must be an RNNCell, saw type: %s" % type(cell).__name__)
		if not isinstance(attention_mechanism, AttentionMechanism):
			raise TypeError("attention_mechanism must be a AttentionMechanism, saw type: %s"
				% type(attention_mechanism).__name__)
		if cell_input_fn is None:
			cell_input_fn = (lambda inputs, attention: array_ops.concat([inputs, attention], -1))
		else:
			if not callable(cell_input_fn):
				raise TypeError("cell_input_fn must be callable, saw type: %s"
					% type(cell_input_fn).__name__)

		if attention_layer_size is not None:
			self._attention_layer = layers_core.Dense(attention_layer_size, name="attention_layer", use_bias=False)
			self._attention_layer_size = attention_layer_size
		else:
			self._attention_layer = None
			self._attention_layer_size = attention_mechanism.values.get_shape()[-1].value

		self._cell = cell
		self._attention_mechanism = attention_mechanism
		self._cell_input_fn = cell_input_fn
		self._output_attention = output_attention
		self._alignment_history = alignment_history
		with ops.name_scope(name, "AttentionWrapperInit"):
			if initial_cell_state is None:
				self._initial_cell_state = None
			else:
				final_state_tensor = nest.flatten(initial_cell_state)[-1]
				state_batch_size = (
					final_state_tensor.shape[0].value
					or array_ops.shape(final_state_tensor)[0])
				error_message = (
					"When constructing AttentionWrapper %s: " % self._base_name +
					"Non-matching batch sizes between the memory "
					"(encoder output) and initial_cell_state.  Are you using "
					"the BeamSearchDecoder?  You may need to tile your initial state "
					"via the tf.contrib.seq2seq.tile_batch function with argument "
					"multiple=beam_width.")
				with ops.control_dependencies(
					[check_ops.assert_equal(state_batch_size,
											self._attention_mechanism.batch_size,
											message=error_message)]):
					self._initial_cell_state = nest.map_structure(
						lambda s: array_ops.identity(s, name="check_initial_cell_state"),
						initial_cell_state)

	@property
	def output_size(self):
		if self._output_attention:
			return self._attention_layer_size
		else:
			return self._cell.output_size

	@property
	def state_size(self):
		return AttentionWrapperState(
			cell_state=self._cell.state_size,
			time=tensor_shape.TensorShape([]),
			attention=self._attention_layer_size,
			alignments=self._attention_mechanism.alignments_size,
			alignment_history=())  # alignment_history is sometimes a TensorArray

	def zero_state(self, batch_size, dtype):
		with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
			if self._initial_cell_state is not None:
				cell_state = self._initial_cell_state
			else:
				cell_state = self._cell.zero_state(batch_size, dtype)
			error_message = (
				"When calling zero_state of AttentionWrapper %s: " % self._base_name +
				"Non-matching batch sizes between the memory "
				"(encoder output) and the requested batch size.  Are you using "
				"the BeamSearchDecoder?  If so, make sure your encoder output has "
				"been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
				"the batch_size= argument passed to zero_state is "
				"batch_size * beam_width.")
			with ops.control_dependencies(
				[check_ops.assert_equal(batch_size,
										self._attention_mechanism.batch_size,
										message=error_message)]):
				cell_state = nest.map_structure(
					lambda s: array_ops.identity(s, name="checked_cell_state"),
					cell_state)
			if self._alignment_history:
				alignment_history = tensor_array_ops.TensorArray(
					dtype=dtype, size=0, dynamic_size=True)
			else:
				alignment_history = ()
			return AttentionWrapperState(
				cell_state=cell_state,
				time=array_ops.zeros([], dtype=dtypes.int32),
				attention=_zero_state_tensors(self._attention_layer_size, batch_size, dtype),
				alignments=self._attention_mechanism.initial_alignments(batch_size, dtype),
				alignment_history=alignment_history)


	def call(self, inputs, state):
		old_cell_output = state.cell_state[-1]
		alignments = self._attention_mechanism(
			old_cell_output, previous_alignments=state.alignments)

		expanded_alignments = array_ops.expand_dims(alignments, 1)
		attention_mechanism_values = self._attention_mechanism.values
		context = math_ops.matmul(expanded_alignments, attention_mechanism_values)
		context = array_ops.squeeze(context, [1])

		if self._attention_layer is not None:
			attention = self._attention_layer(
				array_ops.concat([old_cell_output, context], 1))
		else:
			attention = context

		if self._alignment_history:
			alignment_history = state.alignment_history.write(
				state.time, alignments)
		else:
			alignment_history = ()

		cell_inputs = self._cell_input_fn(inputs, attention)
		cell_state = state.cell_state
		cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

		cell_batch_size = (
			cell_output.shape[0].value or array_ops.shape(cell_output)[0])
		error_message = (
			"When applying AttentionWrapper %s: " % self.name +
			"Non-matching batch sizes between the memory "
			"(encoder output) and the query (decoder output).  Are you using "
			"the BeamSearchDecoder?  You may need to tile your memory input via "
			"the tf.contrib.seq2seq.tile_batch function with argument "
			"multiple=beam_width.")
		with ops.control_dependencies(
			[check_ops.assert_equal(cell_batch_size,
									self._attention_mechanism.batch_size,
									message=error_message)]):
			cell_output = array_ops.identity(
				cell_output, name="checked_cell_output")

		next_state = AttentionWrapperState(
			time=state.time + 1,
			cell_state=next_cell_state,
			attention=attention,
			alignments=alignments,
			alignment_history=alignment_history)

		if self._output_attention:
			return attention, next_state
		else:
			return cell_output, next_state