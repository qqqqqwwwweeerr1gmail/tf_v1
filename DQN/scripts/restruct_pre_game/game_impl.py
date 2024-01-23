
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from time import time


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
            game_succeed = len(np.where(self.board == 0)[0]) == 0
            if game_succeed:
                return (1, game_succeed)
            else:
                return (0, False)
        else:
            return (-1, True)

if __name__ == '__main__':


    game = Game()
    game.reset()

    eproch = 100

    all_reward = []


    for e in range(eproch):
        game_over = False
        game.reset()
        while not game_over:

            state = np.copy(game.board)
            print(state)
            action = random.randint(0,3)
            print(action)
            reward, game_over = game.play(action)
            print(reward,game_over)
            all_reward.append(reward)

    print(all_reward)
    total_reward = sum(all_reward)
    print(total_reward)









