
class gym_Game:
    board = None
    board_size = 0

    def __init__(self, board_size=4):
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.board = np.zeros(self.board_size)

    reward, game_over = game.play(action)


    def step(self, cell):

        state2, reward, done, info

        # returns a tuple: (reward, game_over?)
        if self.board[cell] == 0:
            self.board[cell] = 1
            game_over = len(np.where(self.board == 0)[0]) == 0
            return (1, game_over)
        else:
            return (-1, False)

game = Game()


import time,random
seed = 1546847731  # or try a new seed by using: seed = int(time())
seed = int(time())
random.seed(seed)
print('Seed: {}'.format(seed))

import tensorflow as tf
import numpy as np


class QNetwork:
    def __init__(self, hidden_layers_size, learning_rate, input_size=4, output_size=4):

        self.num_of_games = 1
        self.cur_of_games = 0
        self.start_epsilon = 0.5
        self.cur_eps = 0.3
        self.gamma = 0.95
        self.batch_size = 10

        self.q_s1a = tf.placeholder(shape=(None, output_size), dtype=tf.float32)
        self.r = tf.placeholder(shape=None, dtype=tf.float32)
        self.state_input = self.s = tf.placeholder(shape=(None, input_size), dtype=tf.float32)
        self.enum_a = tf.placeholder(shape=(None, 2), dtype=tf.int32)
        layer = self.s
        self.layers = []
        self.layers.append(self.s)
        for l in hidden_layers_size:
            layer = tf.layers.dense(inputs=layer, units=l, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed))
            self.layers.append(layer)
        self.q_s = tf.layers.dense(inputs=layer, units=output_size,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        self.layers.append(self.q_s)
        self.q_sa_c = tf.gather_nd(self.q_s, indices=self.enum_a)
        self.q_sa_t = self.r + self.gamma * tf.reduce_max(self.q_s1a, axis=1)
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.q_sa_t, predictions=self.q_sa_c))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def egreedy_action(self, state):
        self.cur_eps = self.start_epsilon * self.cur_of_games / self.num_of_games
        Q_val_output = self.session.run(self.Q_val, feed_dict={self.state_input: [state]})[0]
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)  # 左闭右闭区间，np.random.randint为左闭右开区间
        else:
            return np.argmax(Q_val_output) # random output is depend



from collections import deque

class ReplayMemory:
    memory = None
    c = counter = 0

    def __init__(self, size):
        self.memory_size = 2000
        self.m = deque(maxlen=size)

    def append(self, element):
        self.m.append(element)
        self.c += 1

    def sample(self, n):
        # n = len(self.memory)
        a = random.sample(self.m, n)
        b = self.m
        c = list(self.m)[-n:]
        # print(self.memory)
        d = self.m
        return list(self.m)[-n:]


tf.reset_default_graph()
tf.set_random_seed(seed)
qnn = QNetwork(hidden_layers_size=[20, 20], gamma=gamma, learning_rate=0.01)

memory = ReplayMemory()
sess = tf.Session()
sess.run(tf.global_variables_initializer())


def ag_play(qnn):
    counter = 0
    for g in range(qnn.num_of_games):
        game_over = False
        game.reset()
        total_reward = 0
        while not game_over:
            counter += 1
            state = np.copy(game.board)



















