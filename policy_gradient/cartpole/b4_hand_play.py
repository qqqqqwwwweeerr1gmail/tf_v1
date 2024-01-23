import gym
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from scipy.stats import zscore
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode="rgb_array")

end_game_reward = -100

eproch = 1

images = []

for ep in range(eproch):
    game_over = False
    env.reset()
    states = []
    rewards = []
    actions = []
    steps = 0

    while not game_over:
        steps += 1
        current_state = env.state

        action = np.random.choice(env.action_space.n)
        action = int(input())
        next_state, r, game_over, tras,info = env.step(action)
        print('----------------------------------')
        print(next_state)
        print(r)
        print(game_over)
        print(tras)
        print(info)
        print('----------------------------------')
        if game_over and steps < env._max_episode_steps: r = end_game_reward

        states.append(current_state)
        rewards.append(r)
        actions.append(action)

        # img = plt.imshow(env.render())
        img = env.render()
        # plt.show()
        images.append(img)


import pickle
with open('mp4/images2.pkl', 'wb') as file:
    pickle.dump(images, file)



































