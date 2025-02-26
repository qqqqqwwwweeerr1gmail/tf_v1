import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
torch.manual_seed(42)

# Predefined 3-neuron activation vectors for each category (0-9)
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
        # Trainable "mind" with a hidden layer
        self.mind_hidden = nn.Linear(1, 10)
        self.mind_output = nn.Linear(10, 3)

        # Fixed "hands" layer: approximate mapping to category numbers
        self.hands = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            # Adjusted weights to better approximate 0-9 (not perfect, but closer)
            self.hands.weight = nn.Parameter(torch.tensor([[2.0, 4.0, 6.0]]), requires_grad=False)

    def forward(self, x):
        hidden = torch.relu(self.mind_hidden(x))
        control = torch.sigmoid(self.mind_output(hidden))  # 3 control neurons in [0, 1]
        output = self.hands(control)  # Maps to approximate category number
        return output, control


# Initialize the model and optimizer
model = MotivationNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Loss function: minimize distance to predefined vector
def compute_loss(control, target_category):
    target_vector = PREDEFINED_VECTORS[target_category.long()]
    return torch.mean((control - target_vector) ** 2)


# Training loop
def train_model(num_epochs=5000):
    for epoch in range(num_epochs):
        desired = torch.tensor([[float(torch.randint(0, 10, (1,)).item())]], dtype=torch.float32)

        output, control = model(desired)
        loss = compute_loss(control, desired.int())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
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
train_model(num_epochs=5000)
test_model()

# Optional: Inspect the learned weights
print("\nLearned mind hidden weights:", model.mind_hidden.weight.data)
print("Learned mind output weights:", model.mind_output.weight.data)
print("Fixed hands weights:", model.hands.weight.data)























