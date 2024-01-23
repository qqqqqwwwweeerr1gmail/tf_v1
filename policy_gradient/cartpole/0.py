

import gym
import matplotlib.pyplot as plt


env = gym.make('CartPole-v1', render_mode="rgb_array")
env.reset()
# plt.imshow(env.render('rgb_array'))
# #env.render()



prev_screen = env.render()
print(prev_screen)
plt.imshow(prev_screen)
plt.show()


import logging
log_path = './file.log'
logging.basicConfig(filename=log_path, level=logging.DEBUG)
logging.info('12313')
# gym==0.23.1
# gym==0.26.2







