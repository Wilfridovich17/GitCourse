import tensorflow as tf
import numpy as np

from math import pi

grid = np.arange(-3.0, 3.0, 0.1)

mean =  0.0
sigma = 1.0

sess = tf.compat.v1.Session()
with tf.compat.v1.get_default_graph().as_default():
	x = tf.convert_to_tensor(grid, dtype = tf.float32)
	p1 = tf.exp(tf.negative(tf.pow(x - mean, 2.0)) / (2.0 * tf.pow(sigma, 2.0)))
	p2 = 1.0/(sigma * tf.sqrt(2.0 * pi))
	resultado = sess.run(p1)
	
print(resultado)
