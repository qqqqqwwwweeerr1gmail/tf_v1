import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

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
        # Trainable "mind" with two hidden layers
        self.mind_hidden1 = nn.Linear(1, 32)  # First hidden layer
        self.mind_hidden2 = nn.Linear(32, 16)  # Second hidden layer
        self.mind_output = nn.Linear(16, 3)  # Output layer

        # Xavier initialization to break symmetry
        nn.init.xavier_uniform_(self.mind_hidden1.weight)
        nn.init.xavier_uniform_(self.mind_hidden2.weight)
        nn.init.xavier_uniform_(self.mind_output.weight)

        # Fixed "hands" layer (for reference only)
        self.hands = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.hands.weight = nn.Parameter(torch.tensor([[2.0, 4.0, 6.0]]), requires_grad=False)

    def forward(self, x):
        hidden1 = torch.relu(self.mind_hidden1(x))
        hidden2 = torch.relu(self.mind_hidden2(hidden1))
        control = torch.sigmoid(self.mind_output(hidden2))  # 3 control neurons in [0, 1]
        output = self.hands(control)  # Reference output only
        return output, control


# Predict category by finding the most similar predefined vector
def predict_category(control):
    distances = torch.sum((PREDEFINED_VECTORS - control) ** 2, dim=1)  # Euclidean distance
    return torch.argmin(distances).item()  # Index of closest vector


# Initialize the model and optimizer
model = MotivationNN()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  # SGD with momentum
scheduler = lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)  # Slower lr decay


# Loss function: MSE + L1 regularization for sparsity
def compute_loss(control, target_category):
    target_vector = PREDEFINED_VECTORS[target_category.long()]
    mse_loss = torch.mean((control - target_vector) ** 2)
    l1_loss = 0.01 * torch.mean(torch.abs(control))  # Encourage sparsity
    return mse_loss + l1_loss


# Training loop
def train_model(num_epochs=10000):
    for epoch in range(num_epochs):
        desired = torch.tensor([[float(torch.randint(0, 10, (1,)).item())]], dtype=torch.float32)

        output, control = model(desired)
        loss = compute_loss(control, desired.int())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 1000 == 0:
            predicted_category = predict_category(control)
            control_values = control.squeeze().tolist()
            target_vector = PREDEFINED_VECTORS[desired.int().item()].tolist()
            print(f"Epoch {epoch + 1}, Desired: {int(desired.item())}, Predicted: {predicted_category}, "
                  f"Raw Output: {output.item():.2f}, Brain Output: [{control_values[0]:.2f}, "
                  f"{control_values[1]:.2f}, {control_values[2]:.2f}], Target Vector: [{target_vector[0]:.2f}, "
                  f"{target_vector[1]:.2f}, {target_vector[2]:.2f}], Loss: {loss.item():.6f}")


# Test the model
def test_model():
    print("\nTesting trained model:")
    for i in range(10):
        desired = torch.tensor([[float(i)]], dtype=torch.float32)
        output, control = model(desired)
        predicted_category = predict_category(control)
        control_values = control.squeeze().tolist()
        target_vector = PREDEFINED_VECTORS[i].tolist()
        print(f"Desired: {i}, Predicted: {predicted_category}, Raw Output: {output.item():.2f}, "
              f"Brain Output: [{control_values[0]:.2f}, {control_values[1]:.2f}, {control_values[2]:.2f}], "
              f"Target Vector: [{target_vector[0]:.2f}, {target_vector[1]:.2f}, {target_vector[2]:.2f}]")


# Run training and testing
test_model()
train_model(num_epochs=10000)
test_model()

# Optional: Inspect the learned weights
print("\nLearned mind hidden1 weights:", model.mind_hidden1.weight.data)
print("Learned mind hidden2 weights:", model.mind_hidden2.weight.data)
print("Learned mind output weights:", model.mind_output.weight.data)
print("Fixed hands weights:", model.hands.weight.data)






















