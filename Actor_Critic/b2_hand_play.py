
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

while not game_over and not all_r>20:
    steps += 1
    state = np.copy(next_state)

    action = int(input())

    next_state, r, game_over, trans,_ = env.step(action)
    print(next_state, r, game_over, trans,_)
    # env.render()

    img = plt.imshow(env.render())
    img
    print(img)
    plt.text(50, 50, 'Your Text Here', fontsize=12, color='blue')
    plt.show()



    all_r+=r
print('Ended after {} steps'.format(steps))
