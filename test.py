import random
import numpy as np

import tensorflow as tf



a = np.arange(22)
bptt = 5


def gen():
	while True:
		start = random.randint(0, 5)
		print(start)
		for i in range(start, 14, 5):
			yield (a[i:i + 5], a[i+1:i+6])
# for i in gen():
# 	print(i)

a = tf.layers.Dense(1000)
b = a.apply(tf.zeros(shape=(1, 400)))

with tf.variable_scope('dense', reuse=True):
	e = tf.get_variable('kernel')

f = tf.transpose(e)
print(f)

for v in tf.trainable_variables():
	print(v.name)
	print(v.shape)
print(b)