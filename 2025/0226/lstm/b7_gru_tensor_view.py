import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
from torch.utils.tensorboard import SummaryWriter

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


# Define the Neural Network
class MotivationNN(nn.Module):
    def __init__(self, model_path="motivation_model.pth", log_dir="./runs"):
        super(MotivationNN, self).__init__()
        self.mind_hidden1 = nn.Linear(4, 32)  # 4D input to hidden
        self.mind_hidden2 = nn.Linear(32, 16)
        self.mind_output = nn.Linear(16, 3)  # 3D hand command output
        self.model_path = model_path

        # Check if saved model exists
        if not os.path.exists(self.model_path):
            nn.init.xavier_uniform_(self.mind_hidden1.weight)
            nn.init.xavier_uniform_(self.mind_hidden2.weight)
            nn.init.xavier_uniform_(self.mind_output.weight)
            print(f"No saved model found at {self.model_path}. Initialized with Xavier.")
        else:
            self.load_state_dict(torch.load(self.model_path))
            print(f"Loaded saved model from {self.model_path}.")

        # Log the model graph to TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)
        dummy_input = torch.zeros(1, 4)  # Example input (1 batch, 4 dims)
        self.writer.add_graph(self, dummy_input)
        self.writer.flush()
        print(f"Model graph logged to TensorBoard at {log_dir}. Run 'tensorboard --logdir={log_dir}' to view.")

    def forward(self, x):
        hidden1 = torch.relu(self.mind_hidden1(x))
        hidden2 = torch.relu(self.mind_hidden2(hidden1))
        control = torch.sigmoid(self.mind_output(hidden2))  # 3D hand command in [0, 1]
        return control

    def save_model(self):
        torch.save(self.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}.")

    def close_writer(self):
        self.writer.close()


# Predict category by finding the most similar hand command vector
def predict_category(control):
    distances = torch.sum((HAND_COMMAND_VECTORS - control) ** 2, dim=1)
    return torch.argmin(distances).item()


# Loss function
def compute_loss(control, target_category):
    target_vector = HAND_COMMAND_VECTORS[target_category.long()]
    return torch.mean((control - target_vector) ** 2)


# Training function
def train_model(model, num_epochs=10000, force_train=False):
    if os.path.exists(model.model_path) and not force_train:
        print("Model already exists and training not forced. Skipping training.")
        return

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

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

    model.save_model()


# Evaluate the model
def evaluate_model(model):
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


# Main execution
model_path = "motivation_model.pth"
log_dir = "./runs/motivation_model"  # Directory for TensorBoard logs
model = MotivationNN(model_path=model_path, log_dir=log_dir)
train_model(model, num_epochs=10000, force_train=False)
evaluate_model(model)
model.close_writer()  # Clean up TensorBoard writer