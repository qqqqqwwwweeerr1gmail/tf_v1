



from DQN.main.b1_game import game


num_of_games = 2
num_of_games = 2000
num_of_games = 2000
epsilon = 0.3
epsilon = 0.7
gamma = 0.7
gamma = 0.88

import numpy as np
import pandas as pd
import random


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

        done = False
        game.reset()
        total_reward = 0
        state = np.copy(game.board)

        while not done:

            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                st_st = state_to_str(state)
                action = q_table[st_st].idxmax()

            select_Q[all_states.index(state_to_str(state))][action] += 1
            # reward, game_over = game.play(action)


            state1, reward, done, info = game.step(action)

            print(action, state,reward,done)

            total_reward += reward
            if np.sum(game.board) == 4:  # terminal state
                next_state_max_q_value = 0
            else:
                next_state = np.copy(game.board)
                next_state_max_q_value = q_table[state_to_str(next_state)].max()

            q_value_new = reward + gamma * next_state_max_q_value
            q_value_previous_next = q_table.loc[action, state_to_str(state1)]
            if not q_value_previous_next < q_value_new:
                # 永远记住惨痛教训, 不能好了伤疤忘了疼
                q_table.loc[action, state_to_str(state1)] = reward + gamma * next_state_max_q_value

            state = state1
        r_list.append(total_reward)
    # print(q_table.to_list())

    q_table.to_excel('q_table3.xlsx')
    # q_table.to_excel('q_table.xlsx')

if __name__ == '__main__':
    game_play()





















