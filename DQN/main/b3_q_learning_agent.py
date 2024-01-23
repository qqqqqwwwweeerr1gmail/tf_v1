import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from time import time

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import keras
from keras.layers import Dense

from main.b1_game import game


class DQN_agent:
    def __init__(self):
        self.eproch = self.num_of_games = 200
        self.α = self.learning_rate = 0.03
        self.epsilon = 0.3
        self.cur_epsilon = 0.3
        self.γ = 0.9
        self.batch_size = 10
        self.m_size = 2000
        self.m = self.ReplayMemory(self.m_size)
        self.model = self.init_model()
        # self.t_num = 0

        self.a_counter = 0
        self.t_num = 0

    def e_greedy(self,state):
        if random.random() < self.cur_epsilon:
            action = random.randint(0, 3)
        else:
            pred = y_predicted = self.model.predict(np.array([state]), verbose=0)
            action = np.argmax(pred)
        return action


    def init_model(self):
        model = keras.Sequential()
        model.add(Dense(19, input_dim=4, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(19, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(9, activation='tanh', kernel_initializer='he_normal'))
        model.add(Dense(4, activation='sigmoid'))
        # model.add(Dense(4, activation='softmax'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model

    def perceive(self, s, a, reward, s_, done):
        # cur_action = self.action_list[action: action + 1]
        # self.replay_memory_store.append((s, cur_action[0], reward, s_, done))
        # self.replay_memory_store.append((s, cur_action[0], reward, s_, done))
        self.m.append((s, a, reward, s_, done))

        if len(self.m.m) > self.m_size:
            self.m.popleft()  # based on deque
        # 如果超过记忆的容量，则将最久远的记忆移除。
        # 移去并且返回一个元素，deque, 最左侧的那一个
        if self.a_counter % self.batch_size == 0 :
            cost, accuracy = self.train()

            return cost, accuracy
        return None, None

    def train(self):
        self.t_num += 1
        mini_batch = random.sample(self.m.m, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        action_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]
        next_state_batch = [data[3] for data in mini_batch]
        return None,None
        #TODO

    class ReplayMemory:
        m = None
        counter = 0

        def __init__(self, size):
            self.m = deque(maxlen=size)

        def append(self, element):
            self.m.append(element)
            self.counter += 1

        def sample(self, n):
            return random.sample(self.m, n)


dqn_ag = DQN_agent()










