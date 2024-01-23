import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as wd

import pickle

file_rls = r'./video/b5.pkl'
with open(file_rls, 'rb') as file:
    data = my_list = pickle.load(file)

print(len(my_list))

# my_list = [x for j in my_list for x in j ]
x = my_list

# x = [image_array]
fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion()
fig.show()

text = ''
for i in range(len(x)):
    viewer.clear()
    if i % 2 ==0 :
        if text.startswith('left'):
            text += '1'
        else:
            text = 'left'
        plt.text(50, 50, text, fontsize=12, color='blue')
    if i % 2 ==1 :
        if text.startswith('right'):
            text += '1'
        else:
            text = 'right'
        plt.text(50, 50, text, fontsize=12, color='blue')
    viewer.imshow(x[i])

    plt.pause(.1)
    fig.canvas.draw()















































