
import gym
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from scipy.stats import zscore

env = gym.make('CartPole-v1')

end_game_reward = -100

class PolicyGradient:
    def __init__(self, state_size, num_of_actions, hidden_layers, learning_rate):
        self.states = tf.placeholder(shape=(None, state_size), dtype=tf.float32, name='input_states')
        self.acc_r = tf.placeholder(shape=None, dtype=tf.float32, name='accumalated_rewards')
        self.actions = tf.placeholder(shape=None, dtype=tf.int32, name='actions')
        layer = self.states
        for i in range(len(hidden_layers)):
            layer = tf.layers.dense(inputs=layer, units=hidden_layers[i], activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name='hidden_layer_{}'.format(i+1))
        self.last_layer = tf.layers.dense(inputs=layer, units=num_of_actions, activation=tf.nn.tanh,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='output')
        self.action_prob = tf.nn.softmax(self.last_layer)
        self.log_policy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.last_layer, labels=self.actions)
        self.cost = tf.reduce_mean(self.acc_r * self.log_policy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


hidden_layers = [12,12]
gamma = 0.99
learning_rate = 0.001

pg = PolicyGradient(state_size=env.observation_space.shape[0], num_of_actions=env.action_space.n,
                    hidden_layers=hidden_layers, learning_rate=learning_rate)


def print_stuff(s, every=100):
    if game % every == 0 or game == 1:
        print(s)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
data = pd.DataFrame(columns=['game', 'steps', 'cost'])

for g in range(1500):
    game = g + 1
    game_over = False
    env.reset()
    states = []
    rewards = []
    actions = []
    steps = 0

    print_stuff('Starting game {}'.format(game))
    while not game_over:
        steps += 1
        current_state = env.state
        probs = sess.run(pg.action_prob, feed_dict={pg.states: np.expand_dims(current_state, axis=0)}).flatten()
        action = np.random.choice(env.action_space.n, p=probs)
        next_state, r, game_over, _,_ = env.step(action)
        if game_over and steps < env._max_episode_steps: r = end_game_reward

        # Save to memory:
        states.append(current_state)
        rewards.append(r)
        actions.append(action)
    print_stuff('Game {g} has ended after {s} steps.'.format(g=game, s=steps))

    discounted_acc_rewards = np.zeros_like(rewards)
    s = 0.0
    for i in reversed(range(len(rewards))):
        s = s * gamma + rewards[i]
        discounted_acc_rewards[i] = s
    discounted_acc_rewards = zscore(discounted_acc_rewards)

    states, discounted_acc_rewards, actions = shuffle(states, discounted_acc_rewards, actions)
    c, _ = sess.run([pg.cost, pg.optimizer], feed_dict={pg.states: states,
                                                        pg.acc_r: discounted_acc_rewards,
                                                        pg.actions: actions})

    print_stuff('Cost: {}\n----------'.format(c))
    data = data.append({'game': game, 'steps': steps, 'cost': c}, ignore_index=True)


data['steps_moving_average'] = data['steps'].rolling(window=50).mean()
ax = data.plot('game','steps_moving_average', figsize=(10,10), legend=False)
ax.set_ylabel('steps_moving_average')


data['cost_moving_average'] = data['cost'].rolling(window=50).mean()
ax = data.plot('game','cost_moving_average', figsize=(10,10), legend=False)
ax.set_ylabel('cost_moving_average')



env.reset()
env.render()
state = env.state
steps = 0
game_over = False
while not game_over:
    steps += 1
    probs = sess.run(pg.action_prob, feed_dict={pg.states: np.expand_dims(state, axis=0)}).flatten()
    action = np.argmax(probs)
    state, _, game_over, _,_ = env.step(action)
    env.render()
end_reason = 'maximum possible steps' if steps == env._max_episode_steps else 'dropped pole or left frame'
print("Game ended after {} steps ({})".format(steps, end_reason))










