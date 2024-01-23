import pickle

# 从文件中加载对象
with open('./ddpg.pickle', 'rb') as file:
    ac = pickle.load(file)

import gym
import numpy as np
import tensorflow as tf
sess = tf.Session()

env = gym.make('MountainCarContinuous-v0', render_mode="rgb_array")

next_state,di = env.reset()
env.render()
game_over = False
steps = 0
ac.reset_noise()
while not game_over:
    steps += 1
    state = np.copy(next_state)
    action = ac.act(sess, state)
    next_state, r, game_over, trans,_ = env.step(action)
    env.render()
print('Ended after {} steps'.format(steps))























