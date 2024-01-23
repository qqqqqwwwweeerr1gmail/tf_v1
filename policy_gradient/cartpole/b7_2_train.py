import numpy as np
import tensorflow as tf


class PolicyGradient:
    def __init__(self, state_size, num_of_actions, hidden_layers, learning_rate):
        state_size = 2  # For example
        learning_rate = 0.01

        self.states = self.inputs = tf.keras.Input(shape=(None, state_size))
        dense_layer = tf.layers.Dense(3, activation='relu',name='dense_layer')(self.inputs)
        x = tf.layers.Dense(4, activation='relu',name='x')(dense_layer)
        self.last_layer = self.outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        self.acc_r = tf.placeholder(shape=None, dtype=tf.float32, name='accumalated_rewards')
        self.actions = tf.placeholder(shape=None, dtype=tf.int32, name='actions')

        self.action_prob = tf.nn.softmax(self.last_layer)

        self.log_policy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.last_layer, labels=self.actions)

        self.cost = tf.reduce_mean(self.acc_r * self.log_policy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


if __name__ == '__main__':
    pg = PolicyGradient('', '', '', '')
    sess = tf.Session()
    discounted_acc_rewards = np.array([1,1])
    actions = [1,2]

    states = [[(1,1),(2,2)],[(1,1),(2,2)]]

    c, _ = sess.run([pg.cost, pg.optimizer], feed_dict={pg.states: states,
                                                    pg.acc_r: discounted_acc_rewards,
                                                    pg.actions: actions})
