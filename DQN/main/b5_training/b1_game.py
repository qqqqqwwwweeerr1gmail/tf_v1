
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from time import time


class Gym_Game:
    board = None
    board_size = 0

    def __init__(self, board_size=4):
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.board = np.zeros(self.board_size)


    def step(self, action):

        reward = 0
        done = False

        if self.board[action] == 0:
            self.board[action] = 1
            done = game_succeed = len(np.where(self.board == 0)[0]) == 0
            if done:
                reward = 1
            else:
                reward = 1
        else:
            reward = -1

        state2 = np.copy(game.board)
        info = ''

        return state2, reward, done, info

game = Gym_Game()


if __name__ == '__main__':


    game = Gym_Game()
    game.reset()

    eproch = 100

    all_reward = []


    for e in range(eproch):
        game_over = False
        game.reset()
        state = np.copy(game.board)

        while not game_over:
            print(state)
            action = random.randint(0,3)
            print(action)
            # reward, game_over = game.play(action)
            state, reward, done, info = game.step(action)
            game_over = done
            print(reward,game_over)
            all_reward.append(reward)

    print(all_reward)
    total_reward = sum(all_reward)
    print(total_reward)









