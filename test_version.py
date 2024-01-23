import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop


env = gym.make('CartPole-v1')

state = env.reset()
print(state)

env.render()

print('aaaaaa')

import tensorflow as tf
gpu_available = tf.test.is_gpu_available()
print(gpu_available)

ls = tf.config.list_physical_devices('GPU')
print(ls)

import tensorflow as tf
a = tf.test.is_built_with_cuda()
print(a)
a = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
print(a)















