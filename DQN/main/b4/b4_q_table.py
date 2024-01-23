








import pandas as pd
import numpy as np
all_states = ['[0, 0, 0, 0]', '[0, 0, 0, 1]', '[0, 0, 1, 0]', '[0, 0, 1, 1]', '[0, 1, 0, 0]', '[0, 1, 0, 1]',
              '[0, 1, 1, 0]', '[0, 1, 1, 1]', '[1, 0, 0, 0]', '[1, 0, 0, 1]', '[1, 0, 1, 0]', '[1, 0, 1, 1]',
              '[1, 1, 0, 0]', '[1, 1, 0, 1]', '[1, 1, 1, 0]', '[1, 1, 1, 1]']

Q = pd.DataFrame(0, index=np.arange(4), columns=all_states)


def state_to_str(state):
    return str(list(map(int, state.tolist())))


r_list = []  # store the total reward of each game so we can plot it later
print(Q)
















