

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# Build a tensor
tens_1 = tf.constant(12.0)
tens_2 = tf.constant(16.0)
result = tens_1 * tens_2

new_output = tf.compat.v1.Session()
# Display the Content
print(new_output.run(result))





















