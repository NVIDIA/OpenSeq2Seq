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
for i in gen():
	print(i)
