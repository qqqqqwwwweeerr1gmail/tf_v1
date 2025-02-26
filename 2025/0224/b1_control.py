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

        # Fixed "hands" layer: takes 3 neurons and produces 1 output
        # We'll use a simple linear transformation with frozen weights
        self.hands = nn.Linear(3, 1, bias=False)  # No bias for simplicity
        # Set fixed weights for the hands (e.g., [1, 2, 3])
        with torch.no_grad():
            self.hands.weight = nn.Parameter(torch.tensor([[1.0, 2.0, 3.0]]), requires_grad=False)

    def forward(self, x):
        # x is the desired output (e.g., 5)
        control = self.mind(x)  # Get 3 control neurons
        output = self.hands(control)  # Fixed hands operation
        return output, control  # Return both final output and control neurons for inspection


# Initialize the model, optimizer, and loss
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

        # Every 100 epochs, print progress
        if (epoch + 1) % 100 == 0:
            predicted_int = int(output.round().item())
            print(f"Epoch {epoch + 1}, Desired: {int(desired.item())}, Output: {predicted_int}, "
                  f"Raw Output: {output.item():.2f}, Reward: {reward.item():.2f}")


# Test the model after training
def test_model():
    print("\nTesting trained model:")
    for i in range(10):
        desired = torch.tensor([[float(i)]], dtype=torch.float32)
        output, control = model(desired)
        predicted_int = int(output.round().item())
        print(f"Desired: {i}, Predicted: {predicted_int}, Raw Output: {output.item():.2f}")


# Run training and testing

test_model()
train_model(num_epochs=1000)
test_model()

# Optional: Inspect the learned "mind" weights
print("\nLearned mind weights:", model.mind.weight.data)
print("Fixed hands weights:", model.hands.weight.data)























