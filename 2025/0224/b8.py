import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
torch.manual_seed(42)

# Predefined 3-neuron activation vectors for each category (0-9)
# Each vector is in [0, 1], sum <= 3, and unique
PREDEFINED_VECTORS = torch.tensor([
    [0.0, 0.0, 0.0],  # Category 0
    [1.0, 0.0, 0.0],  # Category 1
    [0.0, 1.0, 0.0],  # Category 2
    [0.0, 0.0, 1.0],  # Category 3
    [0.5, 0.5, 0.0],  # Category 4
    [0.0, 0.5, 0.5],  # Category 5
    [0.5, 0.0, 0.5],  # Category 6
    [1.0, 1.0, 0.0],  # Category 7
    [0.0, 1.0, 1.0],  # Category 8
    [1.0, 0.0, 1.0],  # Category 9
], dtype=torch.float32)


# Define the Neural Network
class MotivationNN(nn.Module):
    def __init__(self):
        super(MotivationNN, self).__init__()
        # Trainable "mind" layer: takes desired category (1 number) and produces 3 control neurons
        self.mind = nn.Linear(1, 3)  # 1 input (desired category) -> 3 neurons

        # Fixed "hands" layer: maps predefined vectors to category outputs [0.0, 1.0, ..., 9.0]
        self.hands = nn.Linear(3, 1, bias=False)  # No bias for simplicity
        with torch.no_grad():
            self.hands.weight = nn.Parameter(torch.tensor([[1.5, 3.0, 4.5]]),
                                             requires_grad=False)  # Same weights as before

    def forward(self, x):
        # x is the desired category (e.g., 5)
        control = torch.sigmoid(self.mind(x))  # 3 control neurons, each in [0, 1]
        output = self.hands(control)  # Fixed hands operation, maps to category space
        return output, control  # Return category output and control neurons


# Initialize the model and optimizer
model = MotivationNN()
optimizer = optim.Adam(model.mind.parameters(), lr=0.01)


# Loss function: minimize distance to predefined vector for the desired category
def compute_loss(control, target_category):
    target_vector = PREDEFINED_VECTORS[target_category.long()]  # Get the predefined vector
    return torch.mean((control - target_vector) ** 2)  # Mean squared error to target vector


# Training loop
def train_model(num_epochs=2000):
    for epoch in range(num_epochs):
        # Randomly pick a desired category (0 to 9)
        desired = torch.tensor([[float(torch.randint(0, 10, (1,)).item())]], dtype=torch.float32)

        # Forward pass
        output, control = model(desired)

        # Compute loss based on predefined vector
        loss = compute_loss(control, desired.int())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Every 200 epochs, print progress
        if (epoch + 1) % 200 == 0:
            predicted_int = int(output.round().item())
            control_values = control.squeeze().tolist()
            target_vector = PREDEFINED_VECTORS[desired.int().item()].tolist()
            print(f"Epoch {epoch + 1}, Desired: {int(desired.item())}, Output: {predicted_int}, "
                  f"Raw Output: {output.item():.2f}, Brain Output: [{control_values[0]:.2f}, "
                  f"{control_values[1]:.2f}, {control_values[2]:.2f}], Target Vector: [{target_vector[0]:.2f}, "
                  f"{target_vector[1]:.2f}, {target_vector[2]:.2f}], Loss: {loss.item():.4f}")


# Test the model after training
def test_model():
    print("\nTesting trained model:")
    for i in range(10):
        desired = torch.tensor([[float(i)]], dtype=torch.float32)
        output, control = model(desired)
        predicted_int = int(output.round().item())
        control_values = control.squeeze().tolist()
        target_vector = PREDEFINED_VECTORS[i].tolist()
        print(f"Desired: {i}, Predicted: {predicted_int}, Raw Output: {output.item():.2f}, "
              f"Brain Output: [{control_values[0]:.2f}, {control_values[1]:.2f}, {control_values[2]:.2f}], "
              f"Target Vector: [{target_vector[0]:.2f}, {target_vector[1]:.2f}, {target_vector[2]:.2f}]")


# Run training and testing
test_model()
train_model(num_epochs=2000)
test_model()
train_model(num_epochs=20000)
test_model()

# Optional: Inspect the learned weights
print("\nLearned mind weights:", model.mind.weight.data)
print("Fixed hands weights:", model.hands.weight.data)























