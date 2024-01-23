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

from b1_game import game


class DQN_agent:
    def __init__(self, type='keras'):
        self.eproch = self.num_of_games = 2000
        self.cur_eproch = None
        self.α = self.learning_rate = 0.03
        self.epsilon = 0.5
        self.cur_epsilon = self.epsilon
        self.γ = 0.9
        self.batch_size = 10
        self.mini_batch = None

        self.m_size = 2000
        self.m = self.ReplayMemory(self.m_size)

        self.input_size = 4
        self.output_size = 4
        self.type = type

        if self.type == 'keras':
            self.model = self.init_model_keras()
        # if self.type == 'tf':
        #     self.model = self.DQN_tf()
        # self.t_num = 0

        # for keras encoder
        self.encoder_feature = self.My_encoder_feature_1()
        self.encoder_label = self.My_encoder_label()

        self.a_counter = 0
        self.t_num = 0

    def e_greedy(self, state):
        is_e = False
        if random.random() < self.cur_epsilon:
            action = random.randint(0, 3)
            is_e = True
        else:
            pred = y_predicted = self.model.predict(np.array([state]), verbose=0)
            action = np.argmax(pred)
        return action, is_e

    def init_model_keras(self):
        model = keras.Sequential()
        model.add(Dense(19, input_dim=4, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(19, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(9, activation='tanh', kernel_initializer='he_normal'))
        model.add(Dense(4, activation='sigmoid'))
        # model.add(Dense(4, activation='softmax'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./model',
                                                         save_weights_only=True,
                                                         verbose=1)

        return model

    class My_encoder_feature_1:
        def __init__(self):
            pass

        def is_2d_list(self, fe):
            return all(isinstance(sublist, list) for sublist in fe) and len(fe) > 0 and all(
                isinstance(item, (int, float, str)) for sublist in fe for item in sublist)

        def transform(self, features2d):
            encoded2d = []
            for fe in features2d:

                if isinstance(fe, np.ndarray):
                    fe = fe.tolist()
                    fe = [int(f) for f in fe]

                if self.is_2d_list(fe):
                    fe = fe[0]

                if not isinstance(fe, str):
                    fe = str(fe)

                encode = []
                if eval(fe[1]) == 0:
                    encode.append(np.float64(0.0))
                elif eval(fe[1]) == 1:
                    encode.append(np.float64(1.0))
                if eval(fe[4]) == 0:
                    encode.append(np.float64(0.0))
                elif eval(fe[4]) == 1:
                    encode.append(np.float64(1.0))
                if eval(fe[7]) == 0:
                    encode.append(np.float64(0.0))
                elif eval(fe[7]) == 1:
                    encode.append(np.float64(1.0))
                if eval(fe[10]) == 0:
                    encode.append(np.float64(0.0))
                elif eval(fe[10]) == 1:
                    encode.append(np.float64(1.0))
                encode = np.array(encode)
                encoded2d.append(encode)
            encoded2d = np.array(encoded2d)
            return encoded2d

    class My_encoder_label:
        def __init__(self):
            pass

        # def is_2d_list(self, fe):
        #     return all(isinstance(sublist, list) for sublist in fe) and len(fe) > 0 and all(
        #         isinstance(item, (int, float, str)) for sublist in fe for item in sublist)

        def transform(self, features2d):
            encoded2d = []
            for fe in features2d:

                if isinstance(fe, list):
                    fe = fe[0]

                val = eval(str(fe))
                encode = np.array([np.float64(0.0) if v != val else np.float64(1.0) for v in range(4)])
                encoded2d.append(encode)
            encoded2d = np.array(encoded2d)

            return encoded2d

    class DQN_tf:
        def __init__(self, hidden_layers_size=[20, 20]):
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

    def perceive(self, s, a, reward, s_, done):
        # cur_action = self.action_list[action: action + 1]
        # self.replay_memory_store.append((s, cur_action[0], reward, s_, done))
        # self.replay_memory_store.append((s, cur_action[0], reward, s_, done))
        self.m.append((s, a, reward, s_, done))

        if len(self.m.m) > self.m_size:
            self.m.popleft()  # based on deque
        # 如果超过记忆的容量，则将最久远的记忆移除。
        # 移去并且返回一个元素，deque, 最左侧的那一个
        if self.a_counter % self.batch_size == 0:

            self.cur_epsilon = self.epsilon * (1 - self.cur_eproch / self.eproch)
            self.t_num += 1

            self.mini_batch = random.sample(self.m.m, self.batch_size)
            s = [data[0] for data in self.mini_batch]
            a = [data[1] for data in self.mini_batch]
            r = [data[2] for data in self.mini_batch]
            s_ = [data[3] for data in self.mini_batch]

            if self.type == 'keras':
                cost, accuracy = self.train_keras(s,a,r,s_)
            if self.type == 'tf':
                cost, accuracy = self.train_tf()

            return cost, accuracy
        return None, None

    def train_tf(self):
        self.t_num += 1

        return None, None
        # TODO

    def train_keras(self,s,a,r,s_):



        # print(s)
        # print(a)
        X_train_enc = self.encoder_feature.transform(s)
        # X_test_enc = self.encoder_feature.transform(X_test)
        y_train_enc = self.encoder_label.transform(a)
        # y_test_enc = self.encoder_label.transform(y_test)

        self.model.fit(X_train_enc, y_train_enc, epochs=1, batch_size=self.batch_size, verbose=0)
        # _, accuracy = model.evaluate(X_test_enc, y_test_enc, verbose=1)

        import time
        self.model.save('./model/b5_keras_'+str(time.time())+'.h5')

        return None, None
        # TODO

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
