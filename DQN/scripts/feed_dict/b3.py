

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

tens = tf.compat.v1.placeholder(tf.int32, shape=[2,5])
with tf.compat.v1.Session() as val:
    for i in range(2):
        print(val.run(tens[i][0], feed_dict={tens : [[12,27,95,13,4],[12,27,95,13,8]]}))





















