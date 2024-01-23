
import numpy as np

from b5_q_learning_agent import dqn_ag
from b1_game import game
from b5_q_table import Q, state_to_str


def play_a(ag,game,Q):

    r_list = []
    c_list = []  # same as r_list, but for the cost

    counter = 0  # will be used to trigger network training

    for ep in range(ag.eproch):
        ag.cur_eproch = ep
        done = False
        game.reset()
        # total_reward = 0


        while not done:
            ag.a_counter += 1
            # action = ag.max_action(state)
            s = np.copy(game.board)
            a,is_e = ag.e_greedy(s)
            s_, r, done, info = game.step(a)
            # total_reward += reward
            s_ = np.copy(game.board)

            cost, accuracy = ag.perceive(s, a, r, s_, done)


            Q.loc[a, state_to_str(s)] = r + ag.α * (ag.γ * Q[state_to_str(s_)].max())

    ag.model.save('./model/b5_keras_zfinal.h5')
    Q.to_csv('./model/Q.csv')
    # print('Final cost: {}'.format(c_list[-1]))
    return r_list, c_list

if __name__ == '__main__':


    r_list, c_list = play_a(dqn_ag,game,Q)
    print(dqn_ag.a_counter)
    print(dqn_ag.t_num)

    import pandas as pd
    pd.set_option('max_colwidth', None)
    print(Q)

























