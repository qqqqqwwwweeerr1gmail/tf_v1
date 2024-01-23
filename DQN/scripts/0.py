class QNetwork:
    def __init__(self, hidden_layers_size, gamma, learning_rate, input_size=4, output_size=4):
        self.q_target = tf.placeholder(shape=(None, output_size), dtype=tf.float32)
        self.r = tf.placeholder(shape=None, dtype=tf.float32)
        self.states = tf.placeholder(shape=(None, input_size), dtype=tf.float32)
        self.enum_actions = tf.placeholder(shape=(None, 2), dtype=tf.int32)
        layer = self.states
        for l in hidden_layers_size:
            layer = tf.layers.dense(inputs=layer, units=l, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        self.output = tf.layers.dense(inputs=layer, units=output_size,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        self.predictions = tf.gather_nd(self.output, indices=self.enum_actions)
        self.labels = self.r + gamma * tf.reduce_max(self.q_target, axis=1)
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions))
        self.optimizer = tf.train_keras.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)


qnn = QNetwork(hidden_layers_size=[20,20], gamma=gamma, learning_rate=0.001)
memory = ReplayMemory(memory_size)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

how to view this model 's structure

















