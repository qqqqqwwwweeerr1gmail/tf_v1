
import numpy as np

def play_a(ag,game):

    r_list = []
    c_list = []  # same as r_list, but for the cost

    counter = 0  # will be used to trigger network training

    for ep in range(ag.eproch):
        done = False
        game.reset()
        # total_reward = 0
        s = np.copy(game.board)

        while not done:
            ag.a_counter += 1
            # action = ag.max_action(state)
            a = ag.e_greedy(s)
            s, r, done, info = game.step(a)
            # total_reward += reward
            s_ = np.copy(game.board)

            cost, accuracy = ag.perceive(s, a, r, s_, done)

    # print('Final cost: {}'.format(c_list[-1]))
    return r_list, c_list

if __name__ == '__main__':
    from main.b3_q_learning_agent import dqn_ag
    from main.b1_game import game

    r_list, c_list = play_a(dqn_ag,game)
    print(dqn_ag.a_counter)
    print(dqn_ag.t_num)

























