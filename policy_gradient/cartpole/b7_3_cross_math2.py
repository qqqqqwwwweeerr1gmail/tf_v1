import tensorflow as tf

labels_sparse = [0, 2, 1]
# labels_sparse = [1,0]
# labels_sparse = [0]
# 索引，即真实的类别
# 0表示第一个样本的类别属于第1类；
# 2表示第二个样本的类别属于第3类；
# 1表示第三个样本的类别属于第2类；

logits = tf.constant(value=[[3, 1, -3,1], [1, 4, 3,10], [2, 7, 5,3]],
                     dtype=tf.float32)
# [0.24149366 7.003508   0.14875525]

logits = tf.constant(value=[[1,3,3,3,3],[1,2,3,3,3],[-1,2,3,3,3]],
                     dtype=tf.float32)
# [3.4195685 1.2536811 2.219707]
logits = tf.constant(value=[[1,3,3,3,3]],
                     dtype=tf.float32)
# [3.4195685]

logits = tf.constant(value=[[3,1,-3],[3,1,-3],[3,1,-3]],
                     dtype=tf.float32)
# [1.867786   0.16778599]


loss_sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=labels_sparse,
    logits=logits)

with tf.compat.v1.Session() as sess:
    print("loss_sparse: \n", sess.run(loss_sparse))


softmax_logit = tf.nn.softmax(logits)
print(softmax_logit)

labels_sparse = [0]

softmax_cross_entropy_logit = -(tf.math.log(softmax_logit[labels_sparse]))
print(softmax_cross_entropy_logit)


with tf.compat.v1.Session() as sess:
    print("softmax_logit: \n", sess.run(softmax_logit))
    print("softmax_cross_entropy_logit: \n", sess.run(softmax_cross_entropy_logit))








