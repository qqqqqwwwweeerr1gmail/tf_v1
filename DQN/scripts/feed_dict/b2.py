import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
# Build a graph
graph = tf.compat.v1.Graph()
with graph.as_default():
    # declare a placeholder that is 3 by 4 of type float32
    input_tens = tf.compat.v1.placeholder(tf.int32, shape=(2, 2), name='input_tensor')

    # Perform some operation on the placeholder
    result = input_tens * 3

# Create an input array to be fed
# arr = np.ones((2, 2))
arr = np.array([[1,1],[1,1]])
arr = np.zeros((2, 2))

# Create a session, and run the graph
with tf.compat.v1.Session(graph=graph) as val:
    # run the session up to node b, feeding an array of values into a
    new_output = val.run(result, feed_dict={input_tens: arr})
    print(new_output)























