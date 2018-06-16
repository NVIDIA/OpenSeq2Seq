# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf

from .encoder import Encoder

def conv_bn_actv(layer, name, inputs, filters, kernel_size, activation_fn, strides,
                   padding, regularizer, training, data_format, bn_momentum,
                   bn_epsilon):
	"""Helper function that applied convolution, batch norm and activation."""
	conv = layer(
		name="{}".format(name),
		inputs=inputs,
		filters=filters,
		kernel_size=kernel_size,
		strides=strides,
		padding=padding,
		kernel_regularizer=regularizer,
		use_bias=False,
		data_format=data_format,
		)
	conv = tf.expand_dims(conv, axis=-1)
	bn = tf.layers.batch_normalization(
    name="{}/bn".format(name),
    inputs=conv,
    gamma_regularizer=regularizer,
    training=training,
    axis=-1 if data_format == 'channels_last' else 1,
    momentum=bn_momentum,
    epsilon=bn_epsilon,
  	)
	bn = tf.squeeze(bn, axis=-1)
	output = activation_fn(bn)
	return output

def conv_wn_actv(layer, name, inputs, filters, kernel_size, activation_fn, strides,
                   padding, regularizer, training, data_format, bn_momentum,
                   bn_epsilon):
	"""Helper functtion that applies weight normalization, convolution and activation."""
	#To-Do how to process dtype = float32 as parameters
	with tf.variable_scope(name):
		in_size_index = -1 if data_format == 'channels_last' else 1
		in_dim = int(inputs.get_shape()[in_size_index])
		out_dim = filters
		#V = tf.get_variable('_V', shape=kernel_size+[in_dim, out_dim], initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(4.0/((sum(kernel_size)/len(kernel_size))*in_dim))), trainable=True)
		V = tf.get_variable('_V', shape=kernel_size+[in_dim, out_dim], initializer=tf.random_normal_initializer(mean=0, stddev=0.01), trainable=True)
		V_norm = tf.norm(V.initialized_value(), axis=[i for i in range(len(V.get_shape()[:-1].as_list()))])  
		g = tf.get_variable('_g', initializer=V_norm, trainable=True)
		b = tf.get_variable('_b', shape=[out_dim], initializer=tf.zeros_initializer(), trainable=True)
		
		#To-Do change this to support all type of convolutions
		W = tf.reshape(g, [1,1,out_dim])*tf.nn.l2_normalize(V,[0,1])
		conv = tf.nn.bias_add(tf.nn.conv1d(name="{}".format(name), value=inputs, filters=W, stride=strides[0], padding=padding), b)
		#conv = tf.nn.conv1d(name="{}".format(name), value=inputs, filters=W, stride=strides[0], padding=padding)
		output = activation_fn(conv)
	return output	

class Wave2LetterEncoder(Encoder):
	"""Wave2Letter like encoder."""
	"""Fully convolutional model"""

	@staticmethod
	def get_required_params():
		return dict(Encoder.get_required_params(), **{
      'dropout_keep_prob': float,
      'convnet_layers': list,
      'activation_fn': None,  # any valid callable
		})

	@staticmethod
	def get_optional_params():
		return dict(Encoder.get_optional_params(), **{
      'data_format': ['channels_first', 'channels_last'],
      'bn_momentum': float,
      'bn_epsilon': float,
      'gated_convolution': bool,
      'weight_normalization' : bool,
		})

	def __init__(self, params, model, name="w2l_encoder", mode='train'):
		"""
		To-Do

		"""
		super(Wave2LetterEncoder, self).__init__(params, model, name, mode)

	def _get_layer(self, layer_type):
		if layer_type == "conv1d":
			return "conv", tf.layers.conv1d
		elif layer_type == "conv2d":
			return "conv", tf.layers.conv2d

	def _encode(self, input_dict):
		"""Creates TensorFlow graph for Wav2Letter like encoder.

		Expects the following inputs::

      input_dict = {
        "src_sequence": tensor of shape [batch_size, sequence length, num features]
        "src_length": tensor of shape [batch_size]
      }
		"""

		source_sequence, src_length = input_dict['source_tensors']

		training = (self._mode == "train")
		dropout_keep_prob = self.params['dropout_keep_prob'] if training else 1.0
		regularizer = self.params.get('regularizer', None)
		data_format = self.params.get('data_format', 'channels_last')
		bn_momentum = self.params.get('bn_momentum', 0.99)
		bn_epsilon = self.params.get('bn_epsilon', 1e-3)

		conv_inputs = source_sequence
		batch_size = conv_inputs.get_shape().as_list()[0]
		if data_format == 'channels_last':
			conv_feats = conv_inputs #B T F
		else:
			conv_feats = tf.transpose(conv_inputs, [0, 2, 1]) #B F T

		# ----- Convolutional layers -----------------------------------------------
		convnet_layers = self.params['convnet_layers']

		for idx_convnet in range(len(convnet_layers)):
			layer_type = convnet_layers[idx_convnet]['type']
			layer_repeat_fixed = convnet_layers[idx_convnet]['repeat']
			layer_repeat_moving = layer_repeat_fixed

			while(layer_repeat_moving != 0):
				layer_repeat_moving = layer_repeat_moving -1
				layer_name, layer = self._get_layer(layer_type)
				if layer_name == "conv":
					ch_out = convnet_layers[idx_convnet]['num_channels']
					conv_block = conv_bn_actv
					if self.params['gated_convolution']: ch_out = 2*ch_out
					if self.params['weight_normalization']: conv_block = conv_wn_actv

					kernel_size = convnet_layers[idx_convnet]['kernel_size']
					strides = convnet_layers[idx_convnet]['stride']
					padding = convnet_layers[idx_convnet]['padding']

					conv_feats = conv_block(
						layer = layer,
						name="conv{}{}".format(idx_convnet + 1, layer_repeat_fixed + 1 - layer_repeat_moving),
						inputs=conv_feats,
						filters=ch_out,
						kernel_size=kernel_size,
						activation_fn=self.params['activation_fn'],
						strides=strides,
						padding=padding,
						regularizer=regularizer,
						training=training,
						data_format=data_format,
						bn_momentum=bn_momentum,
						bn_epsilon=bn_epsilon,
					)

					conv_feats = tf.nn.dropout(x=conv_feats, keep_prob=dropout_keep_prob)
					if self.params['gated_convolution']:
						#conv_feats: B T F
						preactivations, gate_inputs = tf.split(conv_feats, num_or_size_splits=2, axis=2)
						gate_outputs = tf.sigmoid(gate_inputs)
						conv_feats_out = tf.multiply(preactivations, gate_outputs)
					else: conv_feats_out = conv_feats


		if data_format == 'channels_first':
			conv_feats_out = tf.transpose(conv_feats_out, [0, 2, 1])

		outputs = conv_feats_out
		return {
			'outputs': outputs,
			'src_length': src_length,
		}
