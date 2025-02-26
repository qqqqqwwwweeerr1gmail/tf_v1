import torch
import torch.nn as nn

# Define the LSTM cell: input_size=1, hidden_size=1
lstm_cell = nn.LSTMCell(input_size=1, hidden_size=1)
print("LSTM Cell:", lstm_cell)

# Create a proper input tensor for a single time step: (batch_size, input_size)
# Here, batch_size=1, input_size=1
input_tensor = torch.randn(1, 1)  # Shape: (1, 1)
print("Input tensor:", input_tensor)

# Initialize hidden and cell states (required for LSTMCell)
# Shape: (batch_size, hidden_size)
h0 = torch.zeros(1, 1)  # Initial hidden state
c0 = torch.zeros(1, 1)  # Initial cell state
print("Initial hidden state (h0):", h0)
print("Initial cell state (c0):", c0)

# Perform inference by calling the LSTM cell
# Pass input and initial states as a tuple
h1, c1 = lstm_cell(input_tensor, (h0, c0))

# Print the outputs
print("New hidden state (h1):", h1)
print("New cell state (c1):", c1)

# Optional: Demonstrate processing a sequence
print("\nProcessing a sequence of 2 time steps:")
sequence = torch.randn(2, 1, 1)  # Shape: (sequence_length, batch_size, input_size)
h_t, c_t = h0, c0  # Start with initial states
for t in range(sequence.size(0)):  # Iterate over time steps
    h_t, c_t = lstm_cell(sequence[t], (h_t, c_t))
    print(f"Time step {t + 1}:")
    print(f"  Input: {sequence[t]}")
    print(f"  Hidden state: {h_t}")
    print(f"  Cell state: {c_t}")























