import tensorflow as tf
import time
start = time.time()

x1 = tf.constant(5)
x2 = tf.constant(6)
result = tf.mul(x1, x2)

with tf.Session() as sess:
    print sess.run(result)
