import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 3)  # Input layer to 3 neurons
        self.fc2 = nn.Linear(3, 1)  # 3 neurons to output layer

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Activation function
        x = self.fc2(x)
        return x


# Define the environment
class SimpleEnv:
    def __init__(self):
        self.target = np.random.randint(0, 10)  # Random target number

    def reset(self):
        self.target = np.random.randint(0, 10)
        return self.target

    def step(self, action):
        # Calculate reward based on how close the action is to the target
        reward = -abs(self.target - action)
        done = True  # Episode ends after one step
        return reward, done


# Training the neural network with reinforcement learning
def train():
    env = SimpleEnv()
    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_episodes = 1000

    for episode in range(num_episodes):
        target = env.reset()
        target_tensor = torch.tensor([target], dtype=torch.float32)

        # Forward pass
        output = model(target_tensor)
        action = output.item()

        # Calculate reward
        reward, done = env.step(action)

        # Calculate loss (negative reward)
        loss = -reward
        loss_tensor = torch.tensor(loss, requires_grad=True)  # Convert to tensor

        # Backward pass and optimization
        optimizer.zero_grad()
        loss_tensor.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f'Episode {episode}, Target: {target}, Output: {action:.2f}, Reward: {reward:.2f}')


if __name__ == "__main__":
    train()