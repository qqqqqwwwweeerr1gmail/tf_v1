import tensorflow as tf

tf.compat.v1.disable_eager_execution()
input_tens_1 = tf.compat.v1.placeholder(tf.float64)
input_tens_2 = tf.compat.v1.placeholder(tf.float64)
result = tf.math.multiply(input_tens_1, input_tens_2)
with tf.compat.v1.Session() as val:
    new_output=val.run(result, feed_dict={input_tens_1: 56, input_tens_2: 78.0})
    print(new_output)























