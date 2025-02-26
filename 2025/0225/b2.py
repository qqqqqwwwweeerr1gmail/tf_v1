import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Set random seed for reproducibility
torch.manual_seed(42)

# Define action will vectors (4D) and corresponding hand command vectors (3D)
ACTION_WILL_VECTORS = torch.tensor([
    [0.0, 0.0, 0.0, 0.0],  # Operation 0
    [1.0, 0.0, 0.0, 0.0],  # Operation 1
    [0.0, 1.0, 0.0, 0.0],  # Operation 2
    [0.0, 0.0, 1.0, 0.0],  # Operation 3
    [0.5, 0.5, 0.0, 0.0],  # Operation 4
    [0.0, 0.5, 0.5, 0.0],  # Operation 5
    [0.5, 0.0, 0.0, 0.5],  # Operation 6
    [1.0, 1.0, 0.0, 0.0],  # Operation 7
    [0.0, 1.0, 1.0, 0.0],  # Operation 8
    [1.0, 0.0, 1.0, 0.0],  # Operation 9
], dtype=torch.float32)

HAND_COMMAND_VECTORS = torch.tensor([
    [0.0, 0.0, 0.0],  # Operation 0
    [1.0, 0.0, 0.0],  # Operation 1
    [0.0, 1.0, 0.0],  # Operation 2
    [0.0, 0.0, 1.0],  # Operation 3
    [0.5, 0.5, 0.0],  # Operation 4
    [0.0, 0.5, 0.5],  # Operation 5
    [0.5, 0.0, 0.5],  # Operation 6
    [1.0, 1.0, 0.0],  # Operation 7
    [0.0, 1.0, 1.0],  # Operation 8
    [1.0, 0.0, 1.0],  # Operation 9
], dtype=torch.float32)


# Define the Neural Network (no hands layer)
class MotivationNN(nn.Module):
    def __init__(self):
        super(MotivationNN, self).__init__()
        self.mind_hidden1 = nn.Linear(4, 32)  # 4D input to hidden
        self.mind_hidden2 = nn.Linear(32, 16)
        self.mind_output = nn.Linear(16, 3)  # 3D hand command output
        # Xavier initialization
        nn.init.xavier_uniform_(self.mind_hidden1.weight)
        nn.init.xavier_uniform_(self.mind_hidden2.weight)
        nn.init.xavier_uniform_(self.mind_output.weight)

    def forward(self, x):
        hidden1 = torch.relu(self.mind_hidden1(x))
        hidden2 = torch.relu(self.mind_hidden2(hidden1))
        control = torch.sigmoid(self.mind_output(hidden2))  # 3D hand command in [0, 1]
        return control  # Only return control vector


# Predict category by finding the most similar hand command vector
def predict_category(control):
    distances = torch.sum((HAND_COMMAND_VECTORS - control) ** 2, dim=1)
    return torch.argmin(distances).item()


# Initialize model and optimizer
model = MotivationNN()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)


# Loss function
def compute_loss(control, target_category):
    target_vector = HAND_COMMAND_VECTORS[target_category.long()]
    return torch.mean((control - target_vector) ** 2)


# Training loop (no Raw Output in print)
def train_model(num_epochs=10000):
    for epoch in range(num_epochs):
        idx = torch.randint(0, 10, (1,)).item()
        action_will = ACTION_WILL_VECTORS[idx:idx + 1]
        desired = torch.tensor([idx], dtype=torch.long)

        control = model(action_will)
        loss = compute_loss(control, desired)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 1000 == 0:
            predicted_category = predict_category(control)
            control_values = control.squeeze().tolist()
            target_vector = HAND_COMMAND_VECTORS[idx].tolist()
            print(f"Epoch {epoch + 1}, Desired: {idx}, Predicted: {predicted_category}, "
                  f"Brain Output: [{control_values[0]:.2f}, {control_values[1]:.2f}, {control_values[2]:.2f}], "
                  f"Target Vector: [{target_vector[0]:.2f}, {target_vector[1]:.2f}, {target_vector[2]:.2f}], "
                  f"Loss: {loss.item():.6f}")


# Evaluate the model (no Raw Output in print)
def evaluate_model():
    print("\nEvaluating trained model:")
    for i in range(10):
        action_will = ACTION_WILL_VECTORS[i:i + 1]
        control = model(action_will)
        predicted_category = predict_category(control)
        control_values = control.squeeze().tolist()
        target_vector = HAND_COMMAND_VECTORS[i].tolist()
        print(f"Desired: {i}, Predicted: {predicted_category}, "
              f"Brain Output: [{control_values[0]:.2f}, {control_values[1]:.2f}, {control_values[2]:.2f}], "
              f"Target Vector: [{target_vector[0]:.2f}, {target_vector[1]:.2f}, {target_vector[2]:.2f}]")


# Run training and evaluation
evaluate_model()
train_model(num_epochs=10000)
evaluate_model()























