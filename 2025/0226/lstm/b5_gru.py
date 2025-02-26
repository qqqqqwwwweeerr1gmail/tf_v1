import torch
import torch.nn as nn

# Define the GRU cell: input_size=1, hidden_size=1
gru_cell = nn.GRUCell(input_size=1, hidden_size=1)
print("GRU Cell:", gru_cell)

# Create a proper input tensor for a single time step: (batch_size, input_size)
# Here, batch_size=1, input_size=1
input_tensor = torch.randn(1, 1)  # Shape: (1, 1)
print("Input tensor:", input_tensor)

# Initialize hidden state (GRU only has hidden state, no cell state)
# Shape: (batch_size, hidden_size)
h0 = torch.zeros(1, 1)  # Initial hidden state
print("Initial hidden state (h0):", h0)

# Perform inference by calling the GRU cell
# Pass input and initial hidden state (no tuple needed, unlike LSTM)
h1 = gru_cell(input_tensor, h0)

# Print the output
print("New hidden state (h1):", h1)

# Optional: Demonstrate processing a sequence
print("\nProcessing a sequence of 2 time steps:")
sequence = torch.randn(2, 1, 1)  # Shape: (sequence_length, batch_size, input_size)

h_t = h0  # Start with initial hidden state
for t in range(sequence.size(0)):  # Iterate over time steps
    h_t = gru_cell(sequence[t], h_t)
    print(f"Time step {t + 1}:")
    print(f"  Input: {sequence[t]}")
    print(f"  Hidden state: {h_t}")























