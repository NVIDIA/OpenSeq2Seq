import tensorflow as tf

class WeightDropLayerNormBasicLSTMCell(tf.contrib.rnn.RNNCell):
  """LSTM unit with layer normalization and recurrent dropout.
  This class adds layer normalization and recurrent dropout to a
  basic LSTM unit. Layer normalization implementation is based on:
    https://arxiv.org/abs/1607.06450.
  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
  and is applied before the internal nonlinearities.
  Recurrent dropout is base on:
    https://arxiv.org/abs/1603.05118
  "Recurrent Dropout without Memory Loss"
  Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               activation=tf.tanh,
               layer_norm=True,
               norm_gain=1.0,
               norm_shift=0.0,
               recurrent_keep_prob=1.0,
               recurrent_dropout_seed=None,
               weight_keep_prob=1.0,
               weight_dropout_seed=None,
               reuse=None):
    """Initializes the basic LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      activation: Activation function of the inner states.
      layer_norm: If `True`, layer normalization will be applied.
      norm_gain: float, The layer normalization gain initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      norm_shift: float, The layer normalization shift initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
        recurrent dropout probability value. If float and 1.0, no dropout will
        be applied.
      dropout_prob_seed: (optional) integer, the randomness seed.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(WeightDropLayerNormBasicLSTMCell, self).__init__(_reuse=reuse)

    self._num_units = num_units
    self._activation = activation
    self._forget_bias = forget_bias
    self._recurrent_keep_prob = recurrent_keep_prob
    self._recurrent_dropout_seed = recurrent_dropout_seed
    self._weight_keep_prob = weight_keep_prob
    self._weight_dropout_seed = weight_dropout_seed
    self._layer_norm = layer_norm
    self._norm_gain = norm_gain
    self._norm_shift = norm_shift
    self._reuse = reuse

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope, dtype=tf.float32):
    shape = inp.get_shape()[-1:]
    gamma_init = tf.constant_initializer(self._norm_gain)
    beta_init = tf.constant_initializer(self._norm_shift)
    with tf.variable_scope(scope): # replace vs with tf. vs stands for va
      # Initialize beta and gamma for use by layer_norm.
      tf.get_variable("gamma", shape=shape, initializer=gamma_init, dtype=dtype)
      tf.get_variable("beta", shape=shape, initializer=beta_init, dtype=dtype)
    normalized = tf.contrib.layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def _linear(self, args):
    out_size = 4 * self._num_units
    proj_size = args.get_shape()[-1]
    dtype = args.dtype
    print('args shape:', args)

    weights = tf.get_variable("kernel", [proj_size, out_size], dtype=dtype)
    print('weight shape:', weights)
    if (not isinstance(self._weight_keep_prob, float)) or self._weight_keep_prob < 1:
      weights = tf.nn.dropout(weights, self._weight_keep_prob, seed=self._weight_dropout_seed)

    out = tf.matmul(args, weights)
    if not self._layer_norm:
      bias = tf.get_variable("bias", [out_size], dtype=dtype)
      out = tf.nn.bias_add(out, bias)
    return out

  def call(self, inputs, state):
    """LSTM cell with layer normalization and recurrent dropout."""
    c, h = state
    args = tf.concat([inputs, h], 1)
    print('inputs shape:', inputs)
    concat = self._linear(args)
    dtype = args.dtype

    i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)

    if self._layer_norm:
      i = self._norm(i, "input", dtype=dtype)
      j = self._norm(j, "transform", dtype=dtype)
      f = self._norm(f, "forget", dtype=dtype)
      o = self._norm(o, "output", dtype=dtype)

    g = self._activation(j)
    if (not isinstance(self._recurrent_keep_prob, float)) or self._recurrent_keep_prob < 1:
      g = tf.nn.dropout(g, self._recurrent_keep_prob, seed=self._recurrent_dropout_seed)

    new_c = (
        c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * g)
    if self._layer_norm:
      new_c = self._norm(new_c, "state", dtype=dtype)
    new_h = self._activation(new_c) * tf.sigmoid(o)

    new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
    return new_h, new_state


# def test_WeightDropWrapper():
#   cell = WeightDropLayerNormBasicLSTMCell(64)
#   cell.apply(tf.ones(shape=[8, 30]))
  

# test_WeightDropWrapper()