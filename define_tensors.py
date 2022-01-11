import tensorflow as tf
import numpy as np

#Lista de Python
m1 = [[1.0, 2.0], [3.0, 4.0]]

#Array de Numpy
m2 = np.array([[1.0, 2.0], [3.0, 4.0]])

#Constante de Tensorflow
m3 = tf.constant([[1.0, 2.0], [3.0, 4.0]])


t1 = tf.convert_to_tensor(m1, dtype = tf.float32)
t2 = tf.convert_to_tensor(m2, dtype = tf.float32)
t3 = tf.convert_to_tensor(m3, dtype = tf.float32)

print(type(t1))
print(type(t2))
print(type(t3))


print(t1)
print(t2)
print(t3)
