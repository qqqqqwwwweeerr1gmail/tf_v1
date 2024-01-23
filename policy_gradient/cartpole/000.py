







import tensorflow as tf

# Suppose you have a tensor object
tensor = tf.constant([1, 2, 3, 4, 5])

# Initialize a TensorFlow session
with tf.Session() as sess:
    # Run the session to get the value of the tensor
    value = sess.run(tensor)
    print(value)















