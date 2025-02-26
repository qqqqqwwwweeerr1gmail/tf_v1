import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os

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

import torch
import torch.nn as nn
import os

class MotivationNN(nn.Module):
    def __init__(self, model_path):
        super(MotivationNN, self).__init__()
        self.model_path = model_path
        self.mind_hidden1 = nn.Linear(7, 32)  # New input size: 7
        # Define other layers here...

        # Initialize weights
        nn.init.xavier_uniform_(self.mind_hidden1.weight)

        # Load state dict only if it exists and is compatible
        if os.path.exists(self.model_path):
            try:
                self.load_state_dict(torch.load(self.model_path))
                print("Model loaded successfully.")
            except RuntimeError as e:
                print(f"Error loading model: {e}")
                print("Starting with a fresh model instead.")
        else:
            print("No saved model found. Starting fresh.")

# # Usage
# model_path = "F:/tf_v1/2025/0225/new_model.pth"
# model = MotivationNN(model_path=model_path)

    def forward(self, x):
        hidden1 = torch.relu(self.mind_hidden1(x))
        hidden2 = torch.relu(self.mind_hidden2(hidden1))
        # Output is the stimulate output, actual command will be stimulate + previous_state
        control = torch.sigmoid(self.mind_output(hidden2))
        return control

    def save_model(self):
        torch.save(self.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}.")

# Predict category by finding the most similar hand command vector
def predict_category(control):
    distances = torch.sum((HAND_COMMAND_VECTORS - control) ** 2, dim=1)
    return torch.argmin(distances).item()

# Training function
def train_model(model, num_epochs=10000, force_train=False):
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    # Initialize previous state as a 3D zero vector
    previous_state = torch.zeros(1, 3, dtype=torch.float32)

    for epoch in range(num_epochs):
        idx = torch.randint(0, 10, (1,)).item()
        action_will = ACTION_WILL_VECTORS[idx:idx + 1]
        target_vector = HAND_COMMAND_VECTORS[idx]

        # Use the previous state as the input state for this step
        input_state = previous_state
        # Concatenate 4D will with 3D previous state to form 7D input
        input_vector = torch.cat((action_will, input_state), dim=1)
        # Model outputs the stimulate output
        stimulate_output = model(input_vector)
        # Actual command is stimulate output plus the previous state
        actual_command = stimulate_output + input_state

        # Loss is based on the actual command approaching the target vector
        loss = torch.mean((actual_command - target_vector) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update previous_state to the current actual command for the next iteration
        previous_state = actual_command.detach()

        if (epoch + 1) % 1000 == 0:
            predicted_category = predict_category(actual_command)
            stimulate_values = stimulate_output.squeeze().tolist()
            state_values = input_state.squeeze().tolist()
            actual_values = actual_command.squeeze().tolist()
            target_values = target_vector.tolist()
            print(f"Epoch {epoch + 1}, Desired: {idx}, Predicted: {predicted_category}, "
                  f"Last Output (State): [{state_values[0]:.2f}, {state_values[1]:.2f}, {state_values[2]:.2f}], "
                  f"Stimulate Output: [{stimulate_values[0]:.2f}, {stimulate_values[1]:.2f}, {stimulate_values[2]:.2f}], "
                  f"Actual Command: [{actual_values[0]:.2f}, {actual_values[1]:.2f}, {actual_values[2]:.2f}], "
                  f"Target Vector: [{target_values[0]:.2f}, {target_values[1]:.2f}, {target_values[2]:.2f}], "
                  f"Loss: {loss.item():.6f}")

    model.save_model()

# Evaluate the model
def evaluate_model(model):
    print("\nEvaluating trained model:")
    for i in range(10):
        # Reset previous state to zero for each evaluation to test from a neutral state
        previous_state = torch.zeros(1, 3, dtype=torch.float32)
        action_will = ACTION_WILL_VECTORS[i:i + 1]
        input_vector = torch.cat((action_will, previous_state), dim=1)
        stimulate_output = model(input_vector)
        actual_command = stimulate_output + previous_state

        predicted_category = predict_category(actual_command)
        stimulate_values = stimulate_output.squeeze().tolist()
        state_values = previous_state.squeeze().tolist()
        actual_values = actual_command.squeeze().tolist()
        target_vector = HAND_COMMAND_VECTORS[i].tolist()
        print(f"Desired: {i}, Predicted: {predicted_category}, "
              f"Last Output (State): [{state_values[0]:.2f}, {state_values[1]:.2f}, {state_values[2]:.2f}], "
              f"Stimulate Output: [{stimulate_values[0]:.2f}, {stimulate_values[1]:.2f}, {stimulate_values[2]:.2f}], "
              f"Actual Command: [{actual_values[0]:.2f}, {actual_values[1]:.2f}, {actual_values[2]:.2f}], "
              f"Target Vector: [{target_vector[0]:.2f}, {target_vector[1]:.2f}, {target_vector[2]:.2f}]")

# Main execution
model_path = "motivation_model.pth"
model = MotivationNN(model_path=model_path)
evaluate_model(model)
train_model(model, num_epochs=10000, force_train=True)  # Force retraining due to architecture change
evaluate_model(model)
model.save_model()























