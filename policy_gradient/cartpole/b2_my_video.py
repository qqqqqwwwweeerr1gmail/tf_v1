
# from pyvirtualdisplay import Display
# display = Display(visible=False, size=(1400, 900))
# _ = display.start()

import gym
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1",render_mode='rgb_array')
env.reset()
img = plt.imshow(env.render())
img
print(img)
plt.show()

import os
# os.environ["DISPLAY"] = f":{display.display}"

import gym
from gym.wrappers import RecordVideo

env = RecordVideo(gym.make("CartPole-v1"), "./mp4")
env.reset()

for i in range(1_000):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(i+1))
        break

import glob

fo = r'F:\tf_v1\policy_gradient'
name_prefix = r'ws_cartpole'
files = glob.glob(f'{fo}/{env.name_prefix}*.mp4')
files

# from IPython.display import Video
#
# Video(files[0], embed=True)
import cv2

# Replace 'video_path' with the actual path to your video file
video_path = 'your_video_file_path.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Read and display the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Video', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()










