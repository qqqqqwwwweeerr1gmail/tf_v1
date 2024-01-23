import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque

class DQN_tf:
    def __init__(self, hidden_layers_size=[20, 20]):
        self.output_size = 2
        self.input_size = 4
        self.eproch = self.num_of_games = 2000
        self.cur_eproch = None
        self.α = self.learning_rate = 0.03
        self.epsilon = 0.5
        self.cur_epsilon = self.epsilon
        self.γ = 0.9
        self.batch_size = 10
        self.mini_batch = None

        self.m_size = 2000

        seed = 1546847731
        self.q_target = tf.placeholder(shape=(None, self.output_size), dtype=tf.float32)
        self.r = tf.placeholder(shape=None, dtype=tf.float32)
        self.states = tf.placeholder(shape=(None, self.input_size), dtype=tf.float32)
        self.enum_actions = tf.placeholder(shape=(None, 2), dtype=tf.int32)
        layer = self.states
        self.layers = []
        self.layers.append(self.states)
        for l in hidden_layers_size:
            layer = tf.layers.dense(inputs=layer, units=l, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed))
            self.layers.append(layer)
        self.output = tf.layers.dense(inputs=layer, units=self.output_size,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        self.layers.append(self.output)
        self.predictions = tf.gather_nd(self.output, indices=self.enum_actions)
        self.labels = self.r + self.γ * tf.reduce_max(self.q_target, axis=1)
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.α).minimize(self.cost)


state_size = 4
action_size = 2

dqn_tf = DQN_tf()

# Assuming you have a TensorFlow session `sess` running:
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Get all trainable variables in the graph
    trainable_vars = tf.trainable_variables()

    # Iterate and print the variable names and values
    for var in trainable_vars:
        print(var.name, sess.run(var))


















