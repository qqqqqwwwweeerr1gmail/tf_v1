import matplotlib.pyplot as plt
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from scipy.stats import zscore


import pandas as pd

data = {'game': [1, 2, 3, 4, 5], 'steps': [11, 22, 13, 14, 35]}
df = pd.DataFrame(data)
print(df)
# data = data.append({'game': game, 'steps': steps}, ignore_index=True)


# Creating a simple line plot
plt.plot(df)
# plt.show()

ax = df.plot('game','steps_moving_average', figsize=(10,10), legend=False)
ax.set_ylabel('steps_moving_average')

plt.show()



















