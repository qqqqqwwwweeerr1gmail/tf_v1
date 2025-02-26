import torch
import torch.nn as nn
torch.manual_seed(42)

# Define the GRU cell: input_size=1, hidden_size=1
gru_cell = nn.GRUCell(input_size=1, hidden_size=1)
print("GRU Cell:", gru_cell)

# View parameters using state_dict
print("\nInternal Parameters (via state_dict):")
state_dict = gru_cell.state_dict()
# for param_name, param_value in state_dict.items():
#     print(f"{param_name}: {param_value} (shape: {param_value.shape})")

# Direct access to parameters
print("\nInternal Parameters (direct access):")
print(f"weight_ih: {gru_cell.weight_ih} (shape: {gru_cell.weight_ih.shape})")
print(f"weight_hh: {gru_cell.weight_hh} (shape: {gru_cell.weight_hh.shape})")
print(f"bias_ih: {gru_cell.bias_ih} (shape: {gru_cell.bias_ih.shape})")
print(f"bias_hh: {gru_cell.bias_hh} (shape: {gru_cell.bias_hh.shape})")

# Create a proper input tensor for a single time step: (batch_size, input_size)
input_tensor = torch.randn(1, 1)  # Shape: (1, 1)
print("\nInput tensor:", input_tensor)

# Initialize hidden state
h0 = torch.zeros(1, 1)  # Initial hidden state
print("Initial hidden state (h0):", h0)

# Perform inference
h1 = gru_cell(input_tensor, h0)
print("New hidden state (h1):", h1)

# Demonstrate processing a sequence
print("\nProcessing a sequence of 2 time steps:")
sequence = torch.randn(2, 1, 1)  # Shape: (sequence_length, batch_size, input_size)
h_t = h0
for t in range(sequence.size(0)):
    h_t = gru_cell(sequence[t], h_t)
    print(f"Time step {t + 1}:")
    print(f"  Input: {sequence[t]}")
    print(f"  Hidden state: {h_t}")























