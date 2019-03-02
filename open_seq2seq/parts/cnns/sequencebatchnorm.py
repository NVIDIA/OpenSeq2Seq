import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import tf_utils

class MaskedBatchNormalization(tf.layers.BatchNormalization):
  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer=tf.zeros_initializer(),
               gamma_initializer=tf.ones_initializer(),
               moving_mean_initializer=tf.zeros_initializer(),
               moving_variance_initializer=tf.ones_initializer(),
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               renorm=False,
               renorm_clipping=None,
               renorm_momentum=0.99,
               fused=None,
               trainable=True,
               virtual_batch_size=None,
               adjustment=None,
               name=None,
               mask=None,
               **kwargs):
    self.mask = mask
    super(MaskedBatchNormalization, self).__init__(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        fused=fused,
        trainable=trainable,
        virtual_batch_size=virtual_batch_size,
        adjustment=adjustment,
        name=name,
        **kwargs)

  def call(self, inputs, training=False):
    original_training_value = training

    in_eager_mode = context.executing_eagerly()
    if self.virtual_batch_size is not None:
      # Virtual batches (aka ghost batches) can be simulated by reshaping the
      # Tensor and reusing the existing batch norm implementation
      original_shape = [-1] + inputs.shape.as_list()[1:]
      expanded_shape = [self.virtual_batch_size, -1] + original_shape[1:]

      # Will cause errors if virtual_batch_size does not divide the batch size
      inputs = tf.reshape(inputs, expanded_shape)

      def undo_virtual_batching(outputs):
        outputs = tf.reshape(outputs, original_shape)
        return outputs

    if self.fused:
      outputs = self._fused_batch_norm(inputs, training=training)
      if self.virtual_batch_size is not None:
        # Currently never reaches here since fused_batch_norm does not support
        # virtual batching
        outputs = undo_virtual_batching(outputs)
      if not context.executing_eagerly() and original_training_value is None:
        outputs._uses_learning_phase = True  # pylint: disable=protected-access
      return outputs

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.get_shape()
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.axis]
    if self.virtual_batch_size is not None:
      del reduction_axes[1]     # Do not reduce along virtual batch dim

    # Broadcasting only necessary for single-axis batch norm where the axis is
    # not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.axis[0]] = input_shape[self.axis[0]].value
    def _broadcast(v):
      if (v is not None and
          len(v.get_shape()) != ndims and
          reduction_axes != list(range(ndims - 1))):
        return tf.reshape(v, broadcast_shape)
      return v

    scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

    def _compose_transforms(scale, offset, then_scale, then_offset):
      if then_scale is not None:
        scale *= then_scale
        offset *= then_scale
      if then_offset is not None:
        offset += then_offset
      return (scale, offset)

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = tf_utils.constant_value(training)
    if training_value is not False:
      if self.adjustment:
        adj_scale, adj_bias = self.adjustment(tf.shape(inputs))
        # Adjust only during training.
        adj_scale = tf_utils.smart_cond(training,
                                        lambda: adj_scale,
                                        lambda: tf.ones_like(adj_scale))
        adj_bias = tf_utils.smart_cond(training,
                                       lambda: adj_bias,
                                       lambda: tf.zeros_like(adj_bias))
        scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

      # Some of the computations here are not necessary when training==False
      # but not a constant. However, this makes the code simpler.
      keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
      if self.mask is not None:
        x = tf.cast(x=inputs, dtype=tf.float32)
        x_masked = x * self.mask
        mask_count = tf.reduce_sum(self.mask, axis=[0, 1], keepdims=keep_dims)
        mean = tf.reduce_sum(x_masked, axis=reduction_axes, keepdims=keep_dims) / mask_count
        variance = tf.reduce_sum(tf.square(x - mean) * self.mask, axis=reduction_axes, keepdims=keep_dims) / mask_count
      else:
        mean, variance = nn.moments(inputs, reduction_axes, keep_dims=keep_dims)

      moving_mean = self.moving_mean
      moving_variance = self.moving_variance

      mean = tf_utils.smart_cond(training,
                                 lambda: mean,
                                 lambda: moving_mean)
      variance = tf_utils.smart_cond(training,
                                     lambda: variance,
                                     lambda: moving_variance)

      if self.virtual_batch_size is not None:
        # This isn't strictly correct since in ghost batch norm, you are
        # supposed to sequentially update the moving_mean and moving_variance
        # with each sub-batch. However, since the moving statistics are only
        # used during evaluation, it is more efficient to just update in one
        # step and should not make a significant difference in the result.
        new_mean = tf.reduce_mean(mean, axis=1, keepdims=True)
        new_variance = tf.reduce_mean(variance, axis=1, keepdims=True)
      else:
        new_mean, new_variance = mean, variance

      if self.renorm:
        r, d, new_mean, new_variance = self._renorm_correction_and_moments(
            new_mean, new_variance, training)
        # When training, the normalized values (say, x) will be transformed as
        # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
        # = x * (r * gamma) + (d * gamma + beta) with renorm.
        r = _broadcast(tf.stop_gradient(r, name='renorm_r'))
        d = _broadcast(tf.stop_gradient(d, name='renorm_d'))
        scale, offset = _compose_transforms(r, d, scale, offset)

      def _do_update(var, value):
        if in_eager_mode and not self.trainable:
          return

        return self._assign_moving_average(var, value, self.momentum)

      mean_update = tf_utils.smart_cond(
          training,
          lambda: _do_update(self.moving_mean, new_mean),
          lambda: self.moving_mean)
      variance_update = tf_utils.smart_cond(
          training,
          lambda: _do_update(self.moving_variance, new_variance),
          lambda: self.moving_variance)
      if not context.executing_eagerly():
        self.add_update(mean_update, inputs=True)
        self.add_update(variance_update, inputs=True)

    else:
      mean, variance = self.moving_mean, self.moving_variance

    mean = tf.cast(mean, inputs.dtype)
    variance = tf.cast(variance, inputs.dtype)
    if offset is not None:
      offset = tf.cast(offset, inputs.dtype)
    outputs = tf.nn.batch_normalization(inputs,
                                     _broadcast(mean),
                                     _broadcast(variance),
                                     offset,
                                     scale,
                                     self.epsilon)
    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    if self.virtual_batch_size is not None:
      outputs = undo_virtual_batching(outputs)
    if not context.executing_eagerly() and original_training_value is None:
      outputs._uses_learning_phase = True  # pylint: disable=protected-access
    return outputs



def masked_batch_normalization(inputs,
                        axis=-1,
                        momentum=0.99,
                        epsilon=1e-3,
                        center=True,
                        scale=True,
                        beta_initializer=tf.zeros_initializer(),
                        gamma_initializer=tf.ones_initializer(),
                        moving_mean_initializer=tf.zeros_initializer(),
                        moving_variance_initializer=tf.ones_initializer(),
                        beta_regularizer=None,
                        gamma_regularizer=None,
                        beta_constraint=None,
                        gamma_constraint=None,
                        training=False,
                        trainable=True,
                        name=None,
                        reuse=None,
                        renorm=False,
                        renorm_clipping=None,
                        renorm_momentum=0.99,
                        fused=None,
                        virtual_batch_size=None,
                        adjustment=None,
                        mask=None):
  layer = MaskedBatchNormalization(
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=center,
      scale=scale,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      moving_mean_initializer=moving_mean_initializer,
      moving_variance_initializer=moving_variance_initializer,
      beta_regularizer=beta_regularizer,
      gamma_regularizer=gamma_regularizer,
      beta_constraint=beta_constraint,
      gamma_constraint=gamma_constraint,
      renorm=renorm,
      renorm_clipping=renorm_clipping,
      renorm_momentum=renorm_momentum,
      fused=fused,
      trainable=trainable,
      virtual_batch_size=virtual_batch_size,
      adjustment=adjustment,
      name=name,
      mask=mask,
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs, training=training)




class SequenceBatchNormalization(tf.layers.Layer, tf.keras.layers.Layer):
  """Applies layer normalization."""

  def __init__(self, hidden_size, momentum, gamma_regularizer=None):
    super(SequenceBatchNormalization, self).__init__()
    self.hidden_size = hidden_size
    self.momentum = momentum
    self.gamma_regularizer = gamma_regularizer

  def _assign_moving_average(self, variable, value, momentum):
    with tf.name_scope(None, 'AssignMovingAvg',
                        [variable, value, momentum]) as scope:
      with tf.colocate_with(variable):
        decay = tf.convert_to_tensor(1.0 - momentum, name='decay')
        if decay.dtype != variable.dtype.base_dtype:
          decay = tf.cast(decay, variable.dtype.base_dtype)
        update_delta = (variable - tf.cast(value, variable.dtype)) * decay
        return tf.assign_sub(variable, update_delta, name=scope)

  def build(self, _):
    self.scale = tf.get_variable(
        "gamma",
        [self.hidden_size],
         initializer=tf.ones_initializer(dtype=tf.float32),
         regularizer=self.gamma_regularizer,
         trainable=True,
         dtype=tf.float32)
    self.bias = tf.get_variable(
        "beta",
        [self.hidden_size],
         initializer=tf.zeros_initializer(dtype=tf.float32),
         trainable=True,
         dtype=tf.float32)
    self.moving_mean  = tf.get_variable(
        "moving_mean",
        [self.hidden_size],
        initializer=tf.zeros_initializer(dtype=tf.float32),
        trainable=False,
        dtype=tf.float32)
    self.moving_variance  = tf.get_variable(
        "moving_variance",
        [self.hidden_size],
        initializer=tf.ones_initializer(dtype=tf.float32),
        trainable=False,
        dtype=tf.float32)
    self.built = True

  def call(self, inputs, mask, training, epsilon):
    dtype = inputs.dtype
    input_shape = inputs.get_shape()
    inputs = tf.cast(x=inputs, dtype=tf.float32)
    if training:
      if mask is not None:
        x_masked = inputs * mask
        mask_count = tf.reduce_sum(mask, axis=[0, 1], keepdims=True)
        mean = tf.reduce_sum(x_masked, axis=[0, 1], keepdims=True) / mask_count
        variance = tf.reduce_sum(tf.square(inputs - mean) * mask, axis=[0, 1], keepdims=True) / mask_count
      else:
        mean, variance = nn.moments(inputs, [0, 1], keep_dims=True)

      def _do_update(var, value):
        value = tf.squeeze(value, axis=0)
        value = tf.squeeze(value, axis=0)
        return self._assign_moving_average(var, value, self.momentum)

      mean_update = tf_utils.smart_cond(
          training,
          lambda: _do_update(self.moving_mean, mean),
          lambda: self.moving_mean)
      variance_update = tf_utils.smart_cond(
          training,
          lambda: _do_update(self.moving_variance, variance),
          lambda: self.moving_variance)
      self.add_update(mean_update, inputs=True)
      self.add_update(variance_update, inputs=True)

    else:
      mean, variance = self.moving_mean, self.moving_variance

    norm_x = (inputs - mean) * tf.rsqrt(variance + epsilon)
    outputs = norm_x * self.scale + self.bias
    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    return tf.cast(x=outputs, dtype=dtype)
