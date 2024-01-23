






#
#
# rewardss = [25]
# b = rewardss[-1:]
# print(b)
# s = sum(rewardss[-100:])
# print(s)
#
#
#
import math
# b = math.log(0.87887824)*3+math.log(0.11894324)*1+math.log(0.00217852)*-3
# print(b)
#
# b = math.log(0.87887824)*3
# print(b)
#
# b = -math.log(0.87887824)
# print(b)

import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

labels_sparse = [0, 2, 1]
va = [[3,1,-3],[3,1,-3],[3,1,-3]]
#
# logits = tf.constant(value=va,
#                      dtype=tf.float32)
#
#
# softmax_logit = tf.nn.softmax(logits)
# print(softmax_logit)
# with tf.compat.v1.Session() as sess:
#     print("softmax_logit: \n", sess.run(softmax_logit))
#
#     loss_sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(
#         labels=labels_sparse,
#         logits=logits)
#
#     print("loss_sparse: \n", sess.run(loss_sparse))

for vi in range(len(va)):
    inner = []
    for i in va[vi] :
        # if i >0:
        #     print(math.log(i))
        # else :
        #     print(-math.log((-i)))
        print(math.e**i)
        inner.append(math.e**i)
    su = sum(inner)
    outer = []
    for ir in inner:
        outer.append(ir/su)
    print(outer)

    la = labels_sparse[vi]
    sparse_softmax_cross_entropy_with_logits = -math.log(outer[la])
    print("sparse_softmax_cross_entropy_with_logits")
    print(sparse_softmax_cross_entropy_with_logits)



so_lg = [[i for i in vi] for vi in va]

















