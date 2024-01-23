import numpy as np
import tensorflow as tf

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from scipy.stats import zscore


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


if __name__ == '__main__':
    pg = PolicyGradient(4, 4, [3,3], 0.01)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    discounted_acc_rewards = np.array([1.5992543342884444, -0.30273832676010964, -1.1654803667813196, 1.145040354459274, 0.03040088213377727, -0.642641611010582, -1.3432925466149876, -0.8151725912910102, -0.13533168912499052, 0.5177207689304882, 0.3569106207706765, -1.7043232964048047, 0.9905667677694499, -0.4718359405329577, 0.6769228156087018, -0.9894463087459888, 0.19447612767995764, 0.8345328418201329, -1.5229008090732385, 1.297969205282199, 1.4493687675968956])

    actions = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    states = [([ 0.03123684, -0.02859763, -0.03309608,  0.00482076]), (0.009405660369399077, -0.4133859836721523, -0.022678643339432807, 0.4700145058842425), (-0.0709265327910416, -1.3893293629526828, 0.08219158473745358, 1.9445709109452154), (0.03345147682409822, -0.22219559468453565, -0.03928247627254725, 0.26411739890455027), (0.014256442661338162, -0.02388769217635603, -0.02441193516595358, -0.09910878063054251), (-0.011025666491802734, -0.803116780477713, 0.0018309264373337214, 1.0439391162243068), (-0.09871312005009525, -1.5852252724776892, 0.12108300295635789, 2.2615599373755377), (-0.027088002101356992, -0.9982629952442832, 0.02270970876181986, 1.3371962398058581), (0.013778688817811042, -0.2186514224205982, -0.02639411077856443, 0.18577337195658122), (0.023542726218620674, -0.22023178088063683, -0.03124023193621335, 0.22068591543845312), (0.019138090601007938, -0.024677534792036604, -0.026826513627444287, -0.08168546095648149), (-0.15824616697336266, -1.1983821437117428, 0.20648425147264762, 1.7716019189925316), (0.029007564930407508, -0.02653560813953393, -0.03400012829445625, -0.04069224394311777), (0.0011379406959560311, -0.6081803593879382, -0.013278353221747956, 0.7554639829540839), (0.024053774312866805, -0.025552404712306548, -0.02999252523443992, -0.0623853350886715), (-0.04705326200624266, -1.1936635392399466, 0.04945363355793703, 1.6368975589758281), (0.018644539905167207, -0.21940486219145225, -0.028460222846573915, 0.20241438403101675), (0.028476852767616828, -0.22115392273750112, -0.0348139731733186, 0.2410723969439339), (-0.13041762549964903, -1.3914270736856822, 0.16631420170386865, 2.008502488438948), (0.03400454560264908, -0.027653438927543017, -0.0389620276218346, -0.016022432535632736), (0.030664886607354162, 0.16698294976474604, -0.032999669183023383, -0.29811792194056075)]
    current_state = (-0.15824616697336266, -1.1983821437117428, 0.20648425147264762, 1.7716019189925316)

    probs = sess.run(pg.action_prob, feed_dict={pg.states: np.expand_dims(current_state, axis=0)}).flatten()

    c, _ ,input,output,h1,h2= sess.run([pg.cost, pg.optimizer,pg.states,pg.last_layer,tf.get_default_graph().get_tensor_by_name('hidden_layer_1/BiasAdd:0'),tf.get_default_graph().get_tensor_by_name('hidden_layer_2/BiasAdd:0')], feed_dict={pg.states: states,
                                                    pg.acc_r: discounted_acc_rewards,
                                                    pg.actions: actions})
    print(c,_)
    print(pg.states)
    print(input)
    print(output)
    print(h1)
    print(h2)

    layer_by_name = tf.get_default_graph().get_tensor_by_name('hidden_layer_1/BiasAdd:0')
    print(layer_by_name)







