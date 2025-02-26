import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as wd

import pickle

with open('./mp4/images_b5.pkl', 'rb') as file:
    data = my_list = pickle.load(file)

print(len(my_list))

# my_list = [x for j in my_list for x in j ]
x = my_list


# x = [image_array]
fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion()
fig.show()

for i in range(len(x)):
    viewer.clear()
    viewer.imshow(x[i])
    plt.pause(.02)
    fig.canvas.draw()























