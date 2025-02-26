import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
torch.manual_seed(42)


# Define the Neural Network
class MotivationNN(nn.Module):
    def __init__(self):
        super(MotivationNN, self).__init__()
        # Trainable "mind" layer: takes desired output (1 number) and produces 3 control neurons
        self.mind = nn.Linear(1, 3)  # 1 input (desired number) -> 3 neurons

        # Fixed "hands" layer: takes 3 neurons and produces 1 output in [0, 9]
        self.hands = nn.Linear(3, 1, bias=False)  # No bias for simplicity
        with torch.no_grad():
            self.hands.weight = nn.Parameter(torch.tensor([[0.5, 1.0, 1.5]]), requires_grad=False)  # Adjusted weights

    def forward(self, x):
        # x is the desired output (e.g., 5)
        control = self.mind(x)  # Get 3 control neurons (brain output)
        raw_output = self.hands(control)  # Fixed hands operation
        # Normalize to [0, 9] without sigmoid
        output = (raw_output % 10 + 10) % 10  # Ensure output is in [0, 9]
        return output, control  # Return bounded output and control neurons


# Initialize the model and optimizer
model = MotivationNN()
optimizer = optim.Adam(model.mind.parameters(), lr=0.01)  # Only optimize the mind


# Reward function: how close is the output to the desired input?
def compute_reward(predicted, target):
    # Negative absolute difference as a reward (closer = higher reward)
    return -torch.abs(predicted - target)


# Training loop
def train_model(num_epochs=1000):
    for epoch in range(num_epochs):
        # Randomly pick a desired output (0 to 9)
        desired = torch.tensor([[float(torch.randint(0, 10, (1,)).item())]], dtype=torch.float32)

        # Forward pass
        output, control = model(desired)

        # Compute reward (our "loss" to maximize)
        reward = compute_reward(output, desired)
        loss = -reward  # Minimize negative reward (maximize reward)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Every 100 epochs, print progress including brain output
        if (epoch + 1) % 100 == 0:
            predicted_int = int(output.round().item())
            control_values = control.squeeze().tolist()  # Convert tensor to list for readable output
            print(f"Epoch {epoch + 1}, Desired: {int(desired.item())}, Output: {predicted_int}, "
                  f"Raw Output: {output.item():.2f}, Brain Output: [{control_values[0]:.2f}, "
                  f"{control_values[1]:.2f}, {control_values[2]:.2f}], Reward: {reward.item():.2f}")


# Test the model after training
def test_model():
    print("\nTesting trained model:")
    for i in range(10):
        desired = torch.tensor([[float(i)]], dtype=torch.float32)
        output, control = model(desired)
        predicted_int = int(output.round().item())
        control_values = control.squeeze().tolist()  # Convert tensor to list for readable output
        print(f"Desired: {i}, Predicted: {predicted_int}, Raw Output: {output.item():.2f}, "
              f"Brain Output: [{control_values[0]:.2f}, {control_values[1]:.2f}, {control_values[2]:.2f}]")


# Run training and testing
test_model()
train_model(num_epochs=1000)
test_model()

# Optional: Inspect the learned weights
print("\nLearned mind weights:", model.mind.weight.data)
print("Fixed hands weights:", model.hands.weight.data)























