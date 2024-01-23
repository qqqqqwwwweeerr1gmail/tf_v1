

import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

env = gym.make('Acrobot-v1', render_mode="rgb_array")

next_state,di = env.reset()
env.render()
game_over = False
steps = 0
all_r = 0

print(env.action_space)


images = []
count = 0

while not game_over and not all_r>20 and count < 500:
    steps += 1
    state = np.copy(next_state)

    # action = int(input())

    action = 0 if (count/10)%2 ==0 else 2
    action = 0 if state[4]>0 else 2

    next_state, r, game_over, trans,_ = env.step(action)
    # print(next_state, r, game_over, trans,_)
    # env.render()

    img = plt.imshow(env.render())
    img
    # print(img)
    # plt.show()

    img = env.render()

    images.append(img)
    count+=1


    all_r+=r
print('Ended after {} steps'.format(steps))
print('all reward {} steps'.format(all_r))


import pickle
with open('./video/b3.pkl', 'wb') as file:
    pickle.dump(images, file)




















