import os
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import rnn_cell_impl

from .utils import deco_print, create_rnn_cell, getdtype
from .attention_wrapper_utils import GatedAttentionWrapper
from .rnn_utils import dynamic_rnn, bidirectional_dynamic_rnn

class BasicRNet_baseline():
	def __init__(self, model_params, global_step=None, force_var_reuse=False, embedding=None):
		self._model_params = model_params
		num_gpus = self._model_params['num_gpus'] if 'num_gpus' in self._model_params else 1

		self._per_gpu_batch_size = self._model_params['batch_size']
		self._global_batch_size = self._per_gpu_batch_size * num_gpus

		self._embedding_trainable = self._model_params['embedding_trainable'] if 'embedding_trainable' in self._model_params else False
		self._mode = self._model_params['mode']
		self._is_training = (self._mode != 'infer')
		self._loss_factor = self._model_params['loss_factor'] if 'loss_factor' in self._model_params else 1.0
		self._attention_type_gated_layer, self._attention_type_self_matching_layer, self._attention_type_output_layer = self._model_params['attention_type'] if 'attention_type' in self._model_params else ['Bahdanau','Bahdanau','Bahdanau']

		if global_step is not None:
			self.global_step = global_step
		else:
			self.global_step = tf.contrib.framework.get_or_create_global_step()

		self._x_bucket_size = self._model_params['bucket_ctx'][-1]
		self._y_bucket_size = self._model_params['bucket_ques'][-1]

		# placeholders for feeding data
		self.x = tf.placeholder(tf.int32, [self._global_batch_size, self._x_bucket_size])
		self.y = tf.placeholder(tf.int32, [self._global_batch_size, self._y_bucket_size])
		self.x_length = tf.placeholder(tf.int32, [self._global_batch_size])
		self.y_length = tf.placeholder(tf.int32, [self._global_batch_size])

		# below we follow data parallelism for multi-GPU training
		# actual per GPU data feeds
		xs = tf.split(value=self.x, num_or_size_splits=num_gpus, axis=0)
		ys = tf.split(value=self.y, num_or_size_splits=num_gpus, axis=0)
		x_lengths = tf.split(value=self.x_length, num_or_size_splits=num_gpus, axis=0)
		y_lengths = tf.split(value=self.y_length, num_or_size_splits=num_gpus, axis=0)

		self._max_word_len = self._model_params['max_word_len']
		self.x_char = tf.placeholder(tf.int32, [self._global_batch_size, self._x_bucket_size, self._max_word_len])
		self.y_char = tf.placeholder(tf.int32, [self._global_batch_size, self._y_bucket_size, self._max_word_len])
		self.x_char_length = tf.placeholder(tf.int32, [self._global_batch_size, self._x_bucket_size])
		self.y_char_length = tf.placeholder(tf.int32, [self._global_batch_size, self._y_bucket_size])

		xs_char = tf.split(value=self.x_char, num_or_size_splits=num_gpus, axis=0)
		ys_char = tf.split(value=self.y_char, num_or_size_splits=num_gpus, axis=0)
		x_char_lengths = tf.split(value=self.x_char_length, num_or_size_splits=num_gpus, axis=0)
		y_char_lengths = tf.split(value=self.y_char_length, num_or_size_splits=num_gpus, axis=0)

		if self._mode != 'infer':
			self.z_pos = tf.placeholder(tf.int32, [self._global_batch_size, 2])
			z_poses = tf.split(value=self.z_pos, num_or_size_splits=num_gpus, axis=0)
		
		losses = []
		predict = []
		for gpu_ind in range(0, num_gpus):
			with tf.device('/gpu:{}'.format(gpu_ind)), tf.variable_scope(
				name_or_scope=tf.get_variable_scope(),
				initializer = tf.random_uniform_initializer(minval=-0.5,maxval=0.5), 
				reuse=force_var_reuse or (gpu_ind > 0)):
				deco_print('Building graph on GPU:{}'.format(gpu_ind))
				if self._mode != 'infer':
					loss_i, predict_i = self._build_forward_pass_graph(x=xs[gpu_ind], x_length=x_lengths[gpu_ind],
														y=ys[gpu_ind], y_length=y_lengths[gpu_ind],
														x_char=xs_char[gpu_ind], x_char_length=x_char_lengths[gpu_ind],
														y_char=ys_char[gpu_ind], y_char_length=y_char_lengths[gpu_ind],
														z_pos=z_poses[gpu_ind],
														embedding=embedding)
					losses += [loss_i]
					predict += [predict_i]
				else:
					predict_i = self._build_forward_pass_graph(x=xs[gpu_ind], x_length=x_lengths[gpu_ind],
												y=ys[gpu_ind], y_length=y_lengths[gpu_ind],
												x_char=xs_char[gpu_ind], x_char_length=x_char_lengths[gpu_ind],
												y_char=ys_char[gpu_ind], y_char_length=y_char_lengths[gpu_ind],
												embedding=embedding)
					predict += [predict_i]

		max_len = tf.reduce_max([tf.shape(predict_i)[1] for predict_i in predict])
		self.predict = tf.concat([tf.pad(predict_i, [[0,0],[0,max_len-tf.shape(predict_i)[1]]]) for predict_i in predict], 0)

		if self._mode != 'infer':
			self.loss = tf.reduce_mean(losses)
			variables = tf.trainable_variables()
			self.trainable_variables = []
			print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
			deco_print("Trainable variables:")
			total_params = 0
			for var in variables:
				if 'embedding' in var.name and not self._embedding_trainable:
					continue
				self.trainable_variables.append(var)
				var_params = 1
				for dim in var.get_shape():
					var_params *= dim.value
				total_params += var_params
				print('Name: {} and shape: {}'.format(var.name, var.get_shape()))
			deco_print('Total trainable parameters: %d' %total_params)
			print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

			if self._model_params['optimizer'] == 'Momentum':
				optimizer = lambda lr: tf.train.MomentumOptimizer(lr, momentum=0.9)
			elif self._model_params['optimizer'] == 'AdaDelta':
				optimizer = lambda lr: tf.train.AdadeltaOptimizer(lr, rho=0.95, epsilon=1e-06)
			else:
				optimizer = self._model_params['optimizer']

			# learning rate
			if 'use_decay' in self._model_params and self._model_params['use_decay'] == True:
				lr = tf.train.exponential_decay(self._model_params['learning_rate'], self.global_step, self._model_params['decay_steps'], self._model_params['decay_rate'], staircase=True)
			else:
				lr = self._model_params['learning_rate']

			self.train_op = tf.contrib.layers.optimize_loss(
				loss =  self.loss * self._loss_factor,
				global_step = self.global_step,
				learning_rate = lr,
				optimizer = optimizer,
				gradient_noise_scale = None,
				gradient_multipliers = None,
				clip_gradients = None if 'max_grad_norm' not in self._model_params else self._model_params['max_grad_norm'],
				learning_rate_decay_fn = None,
				update_ops = None,
				variables = self.trainable_variables,
				name = "Loss_optimization",
				summaries=["learning_rate", "loss", "gradients", "gradient_norm"],
				colocate_gradients_with_ops = True,
				increment_global_step = True)

	def _build_attention(self, num_units, memory, memory_sequence_length, attention_type='Bahdanau'):
		if attention_type == 'Bahdanau':
			attention = attention_wrapper.BahdanauAttention
		elif attention_type == 'Luong':
			attention = attention_wrapper.LuongAttention

		attention_mechanism = attention(num_units=num_units, memory=memory, memory_sequence_length=memory_sequence_length)
		return attention_mechanism

	def _attentive_bidirectional_rnn(self,
									inputs,
									lengths,
									num_units,
									attention_mechanism=False,
									cell_type='gru',
									num_layers=1,
									dp_input_keep_prob=1.0,
									dp_output_keep_prob=1.0):

		cell_fw = create_rnn_cell(cell_type=cell_type,
								num_units=num_units,
								num_layers=num_layers,
								dp_input_keep_prob=dp_input_keep_prob,
								dp_output_keep_prob=dp_output_keep_prob)

		cell_bw = create_rnn_cell(cell_type=cell_type,
								num_units=num_units,
								num_layers=num_layers,
								dp_input_keep_prob=dp_input_keep_prob,
								dp_output_keep_prob=dp_output_keep_prob)

		if attention_mechanism:
			cell_fw = attention_wrapper.AttentionWrapper(cell=cell_fw, attention_mechanism=attention_mechanism, output_attention=False)
			cell_bw = attention_wrapper.AttentionWrapper(cell=cell_bw, attention_mechanism=attention_mechanism, output_attention=False)

		return bidirectional_dynamic_rnn(
			cell_fw=cell_fw,
			cell_bw=cell_bw,
			inputs=inputs,
			sequence_length=lengths,
			dtype=getdtype())

	def _build_encoder(self, x, x_length, y, y_length, embedding = None):
		emb_size = self._model_params['vocab_embedding_dim']
		self._embedding = tf.get_variable(name='embedding', initializer = embedding, trainable = True, dtype=getdtype())
		self._special_token = tf.get_variable(name='W_special', shape=[4,emb_size], dtype=getdtype())
		self._W = tf.concat([self._special_token, self._embedding], 0)
		self._vocab_size = embedding.shape[0] + 4

		print('vocab_size: %d' %self._vocab_size)
		with tf.variable_scope('Encoder/ctx'):
			self._ctx_W = self._W
			embedded_ctx = tf.nn.embedding_lookup(self._ctx_W, x)
			encoder_ctx_output, encoder_ctx_state = self._attentive_bidirectional_rnn(
				inputs=embedded_ctx,
				lengths=x_length,
				attention_mechanism=False,
				cell_type=self._model_params['encoder_cell_type'],
				num_units=self._model_params['encoder_cell_units'],
				num_layers=self._model_params['encoder_layers'],
				dp_input_keep_prob=self._model_params['encoder_dp_input_keep_prob'],
				dp_output_keep_prob=self._model_params['encoder_dp_output_keep_prob'])
			encoder_ctx_outputs = tf.concat(encoder_ctx_output, 2)

		with tf.variable_scope('Encoder/ques'):
			self._ques_W = self._W
			embedded_ques = tf.nn.embedding_lookup(self._ques_W, y)
			encoder_ques_output, encoder_ques_state = self._attentive_bidirectional_rnn(
				inputs=embedded_ques,
				lengths=y_length,
				attention_mechanism=False,
				cell_type=self._model_params['encoder_cell_type'],
				num_units=self._model_params['encoder_cell_units'],
				num_layers=self._model_params['encoder_layers'],
				dp_input_keep_prob=self._model_params['encoder_dp_input_keep_prob'],
				dp_output_keep_prob=self._model_params['encoder_dp_output_keep_prob'])
			encoder_ques_outputs = tf.concat(encoder_ques_output, 2)

		return encoder_ctx_outputs, encoder_ques_outputs

	def _build_char_encoder(self, x_char, x_char_length, y_char, y_char_length):
		with tf.variable_scope('Encoder/char'):
			##########
			self._char_size = 98 ## need to specify
			##########
			print('char_size: %d' %self._char_size)

			char_emb_size = self._model_params['char_emb_size']
			self._char_W = tf.get_variable(name='W_char', shape=[self._char_size, char_emb_size], dtype=getdtype())

			embedded_ctx_char = tf.nn.embedding_lookup(self._char_W, x_char)
			embedded_ques_char = tf.nn.embedding_lookup(self._char_W, y_char)

			# reshape
			embedded_ctx_char_reshape = tf.reshape(embedded_ctx_char, (self._x_bucket_size * self._model_params['batch_size'], self._max_word_len, char_emb_size))
			embedded_ques_char_reshape = tf.reshape(embedded_ques_char, (self._y_bucket_size * self._model_params['batch_size'], self._max_word_len, char_emb_size))
			ctx_char_length_reshape = tf.reshape(x_char_length, [-1])
			ques_char_length_reshape = tf.reshape(y_char_length, [-1])

			# encode
			with tf.variable_scope('ctx'):
				encoder_ctx_output_char, _ = self._attentive_bidirectional_rnn(
					inputs=embedded_ctx_char_reshape,
					lengths=ctx_char_length_reshape,
					attention_mechanism=False,
					cell_type=self._model_params['encoder_char_cell_type'],
					num_units=self._model_params['encoder_char_cell_units'],
					num_layers=self._model_params['encoder_char_layers'],
					dp_input_keep_prob=self._model_params['encoder_char_dp_input_keep_prob'],
					dp_output_keep_prob=self._model_params['encoder_char_dp_output_keep_prob'])
				encoder_ctx_outputs_char = tf.concat(encoder_ctx_output_char, 2)

			with tf.variable_scope('ques'):
				encoder_ques_output_char, _ = self._attentive_bidirectional_rnn(
					inputs=embedded_ques_char_reshape,
					lengths=ques_char_length_reshape,
					attention_mechanism=False,
					cell_type=self._model_params['encoder_char_cell_type'],
					num_units=self._model_params['encoder_char_cell_units'],
					num_layers=self._model_params['encoder_char_layers'],
					dp_input_keep_prob=self._model_params['encoder_char_dp_input_keep_prob'],
					dp_output_keep_prob=self._model_params['encoder_char_dp_output_keep_prob'])
				encoder_ques_outputs_char = tf.concat(encoder_ques_output_char, 2)
				
			ctx_idx = tf.concat([tf.expand_dims(tf.range(tf.shape(ctx_char_length_reshape)[0]), 1), tf.expand_dims(ctx_char_length_reshape - 1, 1)] , 1)
			ques_idx = tf.concat([tf.expand_dims(tf.range(tf.shape(ques_char_length_reshape)[0]), 1), tf.expand_dims(ques_char_length_reshape - 1, 1)] , 1)
			ctx_n = tf.gather_nd(encoder_ctx_outputs_char, ctx_idx)
			ques_n = tf.gather_nd(encoder_ques_outputs_char, ques_idx)

			ctx_n_reshape = tf.reshape(ctx_n, [self._model_params['batch_size'], -1, self._model_params['encoder_char_cell_units'] * 2])
			ques_n_reshape = tf.reshape(ques_n, [self._model_params['batch_size'], -1, self._model_params['encoder_char_cell_units'] * 2])

			return ctx_n_reshape, ques_n_reshape

	def _build_gated_layer(self, ques_outputs, ques_sequence_length, ctx_outputs, ctx_sequence_length):
		with tf.variable_scope('GatedRNN'):
			attention_depth = self._model_params['gated_attention_layer_size']
			attention_mechanism = self._build_attention(num_units=attention_depth, memory=ques_outputs, memory_sequence_length=ques_sequence_length, attention_type='Bahdanau')

			gate_dim = self.embedded_dim * 2
			def attention_decoder_custom_fn(inputs, attention):
				inputs_layer = layers_core.Dense(gate_dim, dtype=getdtype(), activation=tf.sigmoid, use_bias=False)
				inputs_concat = tf.concat([inputs, attention], -1)
				return inputs_layer(inputs_concat) * inputs_concat

			gated_rnn_cell = create_rnn_cell(cell_type=self._model_params['gated_cell_type'],
											num_units=self._model_params['gated_cell_units'],
											num_layers=self._model_params['gated_layers'],
											dp_input_keep_prob=self._model_params['gated_dp_input_keep_prob'],
											dp_output_keep_prob=self._model_params['gated_dp_output_keep_prob'])
				
			attentive_gated_cell = GatedAttentionWrapper(cell=gated_rnn_cell, 
														attention_mechanism=attention_mechanism,
														cell_input_fn=attention_decoder_custom_fn,
														output_attention=False)
			
			final_outputs, final_state = dynamic_rnn(
				cell=attentive_gated_cell,
				inputs=ctx_outputs,
				sequence_length=ctx_sequence_length,
				dtype=getdtype())

			self._gated_output_dim = self._model_params['gated_cell_units'] # output dimemsion

		return final_outputs

	def _build_self_matching_layer(self, inputs, sequence_length):
		with tf.variable_scope('SelfMatching'):
			attention_depth = self._model_params['self_matching_attention_layer_size']
			if self._attention_type_self_matching_layer == 'Luong':
				attention_depth = self._model_params['self_matching_cell_units']
			attention_mechanism = self._build_attention(num_units=attention_depth, memory=inputs, memory_sequence_length=sequence_length, attention_type=self._attention_type_self_matching_layer)
			final_output, final_state = self._attentive_bidirectional_rnn(
				inputs=inputs,
				lengths=sequence_length,
				attention_mechanism=attention_mechanism,
				cell_type=self._model_params['self_matching_cell_type'],
				num_units=self._model_params['self_matching_cell_units'],
				num_layers=self._model_params['self_matching_layers'],
				dp_input_keep_prob=self._model_params['self_matching_dp_input_keep_prob'],
				dp_output_keep_prob=self._model_params['self_matching_dp_output_keep_prob'])
			self_matching_outputs = tf.concat(final_output, 2)
			
		return self_matching_outputs

	def _build_output_layer_context(self, ques_outputs, ques_sequence_length, ctx_outputs, ctx_sequence_length):
		with tf.variable_scope('Output'):
			attention_depth_ques = self._model_params['output_attention_layer_size_ques']
			if self._attention_type_output_layer == 'Luong':
				attention_depth_ques = self._model_params['ques_param_size']
			attention_mechanism_ques = self._build_attention(num_units=attention_depth_ques, memory=ques_outputs, memory_sequence_length=ques_sequence_length, attention_type=self._attention_type_output_layer)

			self.V_ques = tf.get_variable(name='V_ques', shape=[1, self._model_params['ques_param_size']], dtype=getdtype())
			alignments_ques = attention_mechanism_ques(tf.tile(self.V_ques, [self._model_params['batch_size'],1]), previous_alignments=None)
			expanded_alignments_ques = tf.expand_dims(alignments_ques, 1)
			context = tf.matmul(expanded_alignments_ques, ques_outputs)
			context = tf.squeeze(context, [1])

			attention_depth_ans = self._model_params['output_attention_layer_size_ans']
			if self._attention_type_output_layer == 'Luong':
				attention_depth_ans = self.embedded_dim
			attention_mechanism_ans = self._build_attention(num_units=attention_depth_ans, memory=ctx_outputs, memory_sequence_length=ctx_sequence_length, attention_type=self._attention_type_output_layer)

			output_rnn_cell = create_rnn_cell(cell_type=self._model_params['output_cell_type'],
											num_units=self.embedded_dim,
											num_layers=self._model_params['output_layers'],
											dp_input_keep_prob=self._model_params['output_dp_input_keep_prob'],
											dp_output_keep_prob=self._model_params['output_dp_output_keep_prob'])

			# context as initial state
			if self._model_params['output_cell_type'] == 'gru':
				if self._model_params['output_layers'] == 1:
					initial_state = context
				else:
					initial_state = tuple(context for _ in range(self._model_params['output_layers']))
			elif self._model_params['output_cell_type'] == 'lstm':
				if self._model_params['output_layers'] == 1:
					initial_state = rnn_cell_impl.LSTMStateTuple(tf.zeros_like(context, dtype=getdtype()), context)
				else:
					initial_state = tuple(rnn_cell_impl.LSTMStateTuple(tf.zeros_like(context, dtype=getdtype()), context) for _ in range(self._model_params['output_layers']))


			attentive_output_cell = attention_wrapper.AttentionWrapper(cell=output_rnn_cell,
																	attention_mechanism=attention_mechanism_ans,
																	alignment_history=True,
																	cell_input_fn=lambda _, attention: attention,
																	initial_cell_state=initial_state,
																	output_attention=False)

			final_outputs, final_state = tf.nn.static_rnn(
				cell=attentive_output_cell,
				inputs=[tf.zeros([self._model_params['batch_size'], 1]), tf.zeros([self._model_params['batch_size'], 1])],
				dtype=getdtype())
			
			alignment_history = final_state.alignment_history
			return alignment_history

	def _build_forward_pass_graph(self, 
								x, x_length,
								y, y_length,
								x_char=None, x_char_length=None,
								y_char=None, y_char_length=None,
								z_pos=None,
								embedding=None):
		# build encoder
		encoder_ctx_outputs, encoder_ques_outputs = self._build_encoder(x, x_length, y, y_length, embedding=embedding)
		self.embedded_dim = self._model_params['encoder_cell_units'] * 2

		# build char encoder
		encoder_ctx_outputs_char, encoder_ques_outputs_char = self._build_char_encoder(x_char, x_char_length, y_char, y_char_length)
		encoder_ctx_outputs = tf.concat([encoder_ctx_outputs, encoder_ctx_outputs_char], 2)
		encoder_ques_outputs = tf.concat([encoder_ques_outputs, encoder_ques_outputs_char], 2)
		self.embedded_dim = self.embedded_dim + self._model_params['encoder_char_cell_units'] * 2
		
		# build gated attention rnn
		attentive_gated_rnn_output = self._build_gated_layer(encoder_ques_outputs, y_length, encoder_ctx_outputs, x_length)

		# build self matching attention rnn
		attentive_self_matching_rnn_output = self._build_self_matching_layer(attentive_gated_rnn_output,x_length)

		# build output layer
		alignment_history = self._build_output_layer_context(encoder_ques_outputs, y_length, attentive_self_matching_rnn_output, x_length)

		logits = tf.transpose(alignment_history.stack(), perm=[1,0,2])
		pos_predict = tf.argmax(logits, axis = 2)
		
		if self._mode == 'infer':
			return pos_predict

		idx1 = tf.tile(tf.expand_dims(tf.range(self._model_params['batch_size']),1),[1,2])
		idx2 = tf.tile([[0,1]],[self._model_params['batch_size'],1])
		idx3 = z_pos
		mask = tf.concat([tf.expand_dims(idx, 2) for idx in [idx1, idx2, idx3]], axis=2)
		prob = tf.gather_nd(logits, mask)
		epsilon = tf.convert_to_tensor(10e-8, dtype = prob.dtype.base_dtype)
		prob_clipped = tf.clip_by_value(prob, epsilon, 1-epsilon)
		loss = - tf.reduce_mean(tf.log(prob_clipped))
		return loss, pos_predict
