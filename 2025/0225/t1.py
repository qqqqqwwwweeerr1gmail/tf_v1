
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Set random seed for reproducibility
torch.manual_seed(42)

# Define action will vectors (4D) and initial hand command vectors (3D)
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

INITIAL_HAND_COMMAND_VECTORS = torch.tensor([
    [0.0, 0.0, 0.0],  [1.0, 0.0, 0.0],  [0.0, 1.0, 0.0],  [0.0, 0.0, 1.0],
    [0.5, 0.5, 0.0],  [0.0, 0.5, 0.5],  [0.5, 0.0, 0.5],  [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],  [1.0, 0.0, 1.0]
], dtype=torch.float32)

# Define the Neural Network (no hands layer)
class MotivationNN(nn.Module):
    def __init__(self):
        super(MotivationNN, self).__init__()
        self.mind_hidden1 = nn.Linear(4, 32)
        self.mind_hidden2 = nn.Linear(32, 16)
        self.mind_output = nn.Linear(16, 3)
        nn.init.xavier_uniform_(self.mind_hidden1.weight)
        nn.init.xavier_uniform_(self.mind_hidden2.weight)
        nn.init.xavier_uniform_(self.mind_output.weight)

    def forward(self, x):
        hidden1 = torch.relu(self.mind_hidden1(x))
        hidden2 = torch.relu(self.mind_hidden2(hidden1))
        control = torch.sigmoid(self.mind_output(hidden2))
        return control

# Predict category by finding the most similar hand command vector
def predict_category(control, current_hand_vectors):
    distances = torch.sum((current_hand_vectors - control) ** 2, dim=1)
    return torch.argmin(distances).item()

# Loss function
def compute_loss(control, target_vector):
    return torch.mean((control - target_vector) ** 2)

# Simulate hand vector drift over time
def simulate_hand_drift(hand_vectors, drift_scale=0.05):
    noise = torch.randn_like(hand_vectors) * drift_scale
    new_vectors = torch.clamp(hand_vectors + noise, 0, 1)  # Keep in [0, 1]
    return new_vectors

# Initialize model and optimizer
model = MotivationNN()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

# Initial training
def initial_train_model(num_epochs=10000):
    for epoch in range(num_epochs):
        idx = torch.randint(0, 10, (1,)).item()
        action_will = ACTION_WILL_VECTORS[idx:idx +1]
        desired = torch.tensor([idx], dtype=torch.long)

        control = model(action_will)
        loss = compute_loss(control, INITIAL_HAND_COMMAND_VECTORS[desired])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 1000 == 0:
            predicted = predict_category(control, INITIAL_HAND_COMMAND_VECTORS)
            control_values = control.squeeze().tolist()
            target_vector = INITIAL_HAND_COMMAND_VECTORS[idx].tolist()
            print(f"Epoch {epoch +1}, Desired: {idx}, Predicted: {predicted}, "
                  f"Brain Output: [{control_values[0]:.2f}, {control_values[1]:.2f}, {control_values[2]:.2f}], "
                  f"Target Vector: [{target_vector[0]:.2f}, {target_vector[1]:.2f}, {target_vector[2]:.2f}], "
                  f"Loss: {loss.item():.6f}")

# Online inference and adaptation
def infer_and_adapt(num_steps=100):
    current_hand_vectors = INITIAL_HAND_COMMAND_VECTORS.clone()
    print("\nStarting online inference and adaptation:")
    for step in range(num_steps):
        idx = torch.randint(0, 10, (1,)).item()
        action_will = ACTION_WILL_VECTORS[idx:idx +1]

        control = model(action_will)
        predicted = predict_category(control, current_hand_vectors)

        # Simulate hand drift
        current_hand_vectors = simulate_hand_drift(current_hand_vectors)
        actual_vector = current_hand_vectors[idx]

        # Calculate loss and update model
        loss = compute_loss(control, actual_vector)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            control_values = control.squeeze().tolist()
            target_vector = actual_vector.tolist()
            print(f"Step {step +1}, Desired: {idx}, Predicted: {predicted}, "
                  f"Brain Output: [{control_values[0]:.2f}, {control_values[1]:.2f}, {control_values[2]:.2f}], "
                  f"Current Target: [{target_vector[0]:.2f}, {target_vector[1]:.2f}, {target_vector[2]:.2f}], "
                  f"Loss: {loss.item():.6f}")

# Run initial training and then online adaptation
infer_and_adapt(num_steps=100)
initial_train_model(num_epochs=10000)
infer_and_adapt(num_steps=100)



















