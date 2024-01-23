import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as wd

import pickle

with open('./mp4/images2.pkl', 'rb') as file:
    data = my_list = pickle.load(file)

print(my_list)
print(len(my_list))


x = my_list # Some array of images


# x = [image_array]
fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion() # Turns interactive mode on (probably unnecessary)
fig.show() # Initially shows the figure

for i in range(len(x)):
    viewer.clear() # Clears the previous image
    viewer.imshow(x[i]) # Loads the new image
    plt.pause(.1) # Delay in seconds
    fig.canvas.draw() # Draws the image to the screen























