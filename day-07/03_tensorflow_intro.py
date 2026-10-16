# Install first:  pip install tensorflow
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# tensors: scalar, vector, matrix
scalar = tf.constant(7)
vector = tf.constant([1.0, 2.0, 3.0])
matrix = tf.constant([[1, 2], [3, 4]])
print(scalar.shape, vector.shape, matrix.shape)
print(matrix.dtype, matrix.ndim)

# operations
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
print(tf.add(a, b).numpy())
print(tf.matmul(a, b).numpy())      # [[19. 22.] [43. 50.]]
print(a * 2)
print(tf.reduce_sum(a).numpy(), tf.reduce_mean(a).numpy())

# variables are tensors whose values can change (model weights)
w = tf.Variable([1.0, 2.0])
w.assign_add([0.5, 0.5])
print(w.numpy())                    # [1.5 2.5]

# NumPy interoperability
import numpy as np
t = tf.convert_to_tensor(np.array([[1, 2], [3, 4]]))
print(t.numpy().sum())

# reshape and cast
r = tf.reshape(tf.range(6), (2, 3))
print(r.numpy())
print(tf.cast(r, tf.float32).dtype)
