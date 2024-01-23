
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from time import time


# class Game:
#     board = None
#     board_size = 0
#
#     def __init__(self, board_size=4):
#         self.board_size = board_size
#         self.reset()
#
#     def reset(self):
#         self.board = np.zeros(self.board_size)
#
#     def play(self, cell):
#         # returns a tuple: (reward, game_over?)
#         if self.board[cell] == 0:
#             self.board[cell] = 1
#             game_succeed = len(np.where(self.board == 0)[0]) == 0
#             if game_succeed:
#                 return (1, game_succeed)
#             else:
#                 return (0, False)
#         else:
#             return (-1, True)


class Game:
    board = None
    board_size = 0

    def __init__(self, board_size=4):
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.board = np.zeros(self.board_size)

    def play(self, cell):
        # returns a tuple: (reward, game_over?)
        if self.board[cell] == 0:
            self.board[cell] = 1
            game_over = len(np.where(self.board == 0)[0]) == 0
            return (1, game_over)
        else:
            return (-1, False)

game = Game()




# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf




class QNetwork:
    def __init__(self, hidden_layers_size, gamma, learning_rate, input_size=4, output_size=4):
        self.q_target = tf.placeholder(shape=(None, output_size), dtype=tf.float32)
        self.r = tf.placeholder(shape=None, dtype=tf.float32)
        self.states = tf.placeholder(shape=(None, input_size), dtype=tf.float32)
        self.enum_actions = tf.placeholder(shape=(None, 2), dtype=tf.int32)
        layer = self.states
        self.layers = []
        self.layers.append(self.states)
        for l in hidden_layers_size:
            layer = tf.layers.dense(inputs=layer, units=l, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed))
            self.layers.append(layer)
        self.output = tf.layers.dense(inputs=layer, units=output_size,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        self.layers.append(self.output)
        self.predictions = tf.gather_nd(self.output, indices=self.enum_actions)
        self.labels = self.r + gamma * tf.reduce_max(self.q_target, axis=1)
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)


class ReplayMemory:
    memory = None
    counter = 0

    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def append(self, element):
        self.memory.append(element)
        self.counter += 1

    def sample(self, n):
        # n = len(self.memory)
        a = random.sample(self.memory, n)
        b = self.memory
        c = list(self.memory)[-n:]
        # print(self.memory)
        d = self.memory
        return list(self.memory)[-n:]
        # return random.sample(self.memory, n)


num_of_games = 1000
epsilon = 0.1
gamma = 0.99
batch_size = 10
memory_size = 2000


seed = 1546847731  # or try a new seed by using: seed = int(time())
seed = int(time())
random.seed(seed)
print('Seed: {}'.format(seed))

tf.reset_default_graph()
tf.set_random_seed(seed)
qnn = QNetwork(hidden_layers_size=[20,20], gamma=gamma, learning_rate=0.001)

memory = ReplayMemory(memory_size)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


r_list = []  # 用total reward
c_list = []  # same as r_list, but for the cost

n_ls = []
la_ls = []
pr_ls = []
lys_ls = []
q_ls = []
action_ls = []
r_ls = []   # 用current reward

counter = 0  # will be used to trigger network training

def to_csv(n_ls,action_ls,r_ls, la_ls, pr_ls, lys_ls, q_ls, c_list):
    import pandas as pd

    # Assuming all lists have the same length
    lyst = lys_ls
    # lyst1 = lys_ls
    # lyst = np.stack(lyst1)
    states = [ly[0] for ly in lyst]
    hidden1 = [ly[1] for ly in lyst]
    hidden2 = [ly[2] for ly in lyst]
    output = [ly[3] for ly in lyst]

    # states = lyst[:, 0]
    # hidden1 = lyst[:, 1]
    # hidden2 = lyst[:, 2]
    # output = lyst[:, 3]
    print(la_ls)
    la_ls = [np.array([[ga] for ga in la]) for la in la_ls ]
    print(la_ls)
    pr_ls = [np.array([[ga] for ga in la]) for la in pr_ls ]
    r_ls = [np.array(r).reshape(-1, 1) for r in r_ls]
    # la_ls = np.array(la_ls).reshape(-1, 1)
    # pr_ls = np.array(pr_ls).reshape(-1, 1)
    data = {
        # 'lys': lys_ls,
        'next_state': n_ls,
        'states': states,
        'action': action_ls,
        'predictions': pr_ls,
        'output(Q(S,A))': output,
        'reward': r_ls,
        'labels (Q1(S,A))': la_ls,
        'q_target(Q(S1,A))': q_ls,
        'hidden1': hidden1,
        'hidden2': hidden2,
        'cost': c_list
    }

    # lyst = (np.array(lys_ls)).T

    df = pd.DataFrame(data)
    df.to_csv('output_'+str(num_of_games)+'_'+str(int(time()))+'.csv', index=False)


for g in range(num_of_games):
    game_over = False
    game.reset()
    total_reward = 0
    while not game_over:
        counter += 1
        state = np.copy(game.board)
        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            pred = np.squeeze(sess.run(qnn.output,feed_dict={qnn.states: np.expand_dims(game.board,axis=0)}))
            action = np.argmax(pred)
        reward, game_over = game.play(action)



        total_reward += reward
        next_state = np.copy(game.board)
        memory.append({'state':state,'action':action,'reward':reward,'next_state':next_state,'game_over':game_over})
        if counter % batch_size == 0 or game_over:
            # Network training
            batch = memory.sample(batch_size)
            q_target = sess.run(qnn.output,feed_dict={qnn.states: np.array(list(map(lambda x: x['next_state'], batch)))})
            terminals = np.array(list(map(lambda x: x['game_over'], batch)))
            for i in range(terminals.size):
                if terminals[i]:
                    # Remember we use the network's own predictions for the next state while calculatng loss.
                    # Terminal states have no Q-value, and so we manually set them to 0, as the network's predictions
                    # for these states is meaningless
                    q_target[i] = np.zeros(game.board_size)

            n_s = np.array(list(map(lambda x: x['next_state'], batch)))
            s_s = np.array(list(map(lambda x: x['state'], batch)))
            r_s = np.array(list(map(lambda x: x['reward'], batch)))
            e_s = np.array(list(enumerate(map(lambda x: x['action'], batch))))
            print(e_s)
            la = qnn.labels
            q_s = q_target

            # c_s = np.array(list(map(lambda x: x['state'], batch)))
            _, cost,la,pr,lys = sess.run([qnn.optimizer, qnn.cost,qnn.labels,qnn.predictions,qnn.layers],
                               feed_dict={qnn.states: s_s,
                               qnn.r: r_s,
                               qnn.enum_actions: e_s,
                               qnn.q_target: q_target})

            n_ls.append(n_s)
            action_ls.append(e_s)
            r_ls.append(r_s)
            la_ls.append(la)
            pr_ls.append(pr)
            lys_ls.append(lys)
            q_ls.append(q_s)
            c_list.append(cost)
    r_list.append(total_reward)
to_csv(n_ls,action_ls,r_ls, la_ls, pr_ls, lys_ls, q_ls, c_list)
print('Final cost: {}'.format(c_list[-1]))


from tensorflow.keras.utils import plot_model
import tensorflow as tf

def state_to_str(state):
    return str(list(map(int, state.tolist())))


all_states = list()
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                s = np.array([i, j, k, l])
                all_states.append(state_to_str(s))

# print('All possible states:')
# for s in all_states:
#     print(s)


q_table = pd.DataFrame(0, index=np.arange(4), columns=all_states)

# for i in range(2):
#     for j in range(2):
#         for k in range(2):
#             for l in range(2):
#                 b = np.array([i,j,k,l])
#                 if len(np.where(b == 0)[0]) != 0:
#                     action = q_table[state_to_str(b)].idxmax()
#                     pred = q_table[state_to_str(b)].tolist()
#                     print('board: {b}\tpredicted Q values: {p} \tbest action: {a}\tcorrect action? {s}'
#                           .format(b=b,p=pred,a=action,s=b[action]==0))


# for i in range(2):
#     for j in range(2):
#         for k in range(2):
#             for l in range(2):
#                 b = np.array([i,j,k,l])
#                 if len(np.where(b == 0)[0]) != 0:
#                     action = q_table[state_to_str(b)].idxmax()
#                     pred = q_table[state_to_str(b)].tolist()
#                     print('board: {b}\tpredicted Q values: {p} \tbest action: {a}\tcorrect action? {s}'
#                           .format(b=b,p=pred,a=action,s=b[action]==0))
#

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                b = np.array([i,j,k,l])
                if len(np.where(b == 0)[0]) != 0:
                    pred = np.squeeze(sess.run(qnn.output,feed_dict={qnn.states: np.expand_dims(b,axis=0)}))
                    pred = list(map(lambda x: round(x,3),pred))
                    action = np.argmax(pred)
                    print('board: {b}\tpredicted Q values: {p} \tbest action: {a}\tcorrect action? {s}'
                          .format(b=b,p=pred,a=action,s=b[action]==0))



plt.figure(figsize=(14,7))
plt.plot(range(len(r_list)),r_list)
plt.xlabel('Games played')
plt.ylabel('Reward')
plt.show()




plt.figure(figsize=(14,7))
plt.plot(range(len(c_list)),c_list)
plt.xlabel('Trainings')
plt.ylabel('Cost')
plt.show()


sess.close()  # Don't forget to close tf.session
