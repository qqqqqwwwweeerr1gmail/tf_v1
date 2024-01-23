num_of_games = 2000
epsilon = 0.3
gamma = 0.7

import numpy as np
import pandas as pd
import random


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



all_states = ['[0, 0, 0, 0]', '[0, 0, 0, 1]', '[0, 0, 1, 0]', '[0, 0, 1, 1]', '[0, 1, 0, 0]', '[0, 1, 0, 1]',
              '[0, 1, 1, 0]', '[0, 1, 1, 1]', '[1, 0, 0, 0]', '[1, 0, 0, 1]', '[1, 0, 1, 0]', '[1, 0, 1, 1]',
              '[1, 1, 0, 0]', '[1, 1, 0, 1]', '[1, 1, 1, 0]', '[1, 1, 1, 1]']

q_table = pd.DataFrame(0, index=np.arange(4), columns=all_states)


def state_to_str(state):
    return str(list(map(int, state.tolist())))


r_list = []  # store the total reward of each game so we can plot it later

select_Q = [[0 for j in q_table[i]] for i in q_table]
def game_play():
    for g in range(num_of_games):

        game_over = False
        game.reset()
        total_reward = 0
        while not game_over:
            state = np.copy(game.board)
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                st_st = state_to_str(state)
                action = q_table[st_st].idxmax()

            select_Q[all_states.index(state_to_str(state))][action] += 1
            reward, game_over = game.play(action)
            total_reward += reward
            if np.sum(game.board) == 4:  # terminal state
                next_state_max_q_value = 0
            else:
                next_state = np.copy(game.board)
                next_state_max_q_value = q_table[state_to_str(next_state)].max()
            q_table.loc[action, state_to_str(state)] = reward + gamma * next_state_max_q_value
        r_list.append(total_reward)
    print(q_table)

if __name__ == '__main__':
    game_play()





















