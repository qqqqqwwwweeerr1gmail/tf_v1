import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Initialize the LSTM cell.

        Args:
            input_size (int): Size of the input vector.
            hidden_size (int): Size of the hidden state and cell state.
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight for input: combines weights for all gates (input, forget, cell, output)
        # Shape: (4 * hidden_size, input_size)
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))

        # Weight for hidden state: combines weights for all gates
        # Shape: (4 * hidden_size, hidden_size)
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))

        # Biases for input and hidden contributions
        # Shape: (4 * hidden_size)
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        # Initialize weights and biases
        nn.init.xavier_uniform_(self.weight_ih)  # Xavier initialization for weights
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)  # Biases initialized to zeros
        nn.init.zeros_(self.bias_hh)

    def forward(self, x, hidden=None):
        """
        Forward pass of the LSTM cell.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).
            hidden (tuple, optional): Tuple (h_prev, c_prev) containing previous hidden
                                      and cell states, each of shape (batch_size, hidden_size).
                                      If None, initialized to zeros.

        Returns:
            h_t (Tensor): New hidden state, shape (batch_size, hidden_size).
            c_t (Tensor): New cell state, shape (batch_size, hidden_size).
        """
        # If hidden state is not provided, initialize with zeros
        if hidden is None:
            batch_size = x.size(0)
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_prev, c_prev = hidden

        # Compute gates: [input, forget, cell, output]
        # Shape of gates: (batch_size, 4 * hidden_size)
        gates = (x @ self.weight_ih.t()) + self.bias_ih + (h_prev @ self.weight_hh.t()) + self.bias_hh

        # Split gates into four parts, each of shape (batch_size, hidden_size)
        i, f, g, o = torch.chunk(gates, 4, dim=1)

        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)  # Cell candidate
        o = torch.sigmoid(o)  # Output gate

        # Update cell state and hidden state
        c_t = f * c_prev + i * g  # New cell state
        h_t = o * torch.tanh(c_t)  # New hidden state

        return h_t, c_t


# Example usage
if __name__ == "__main__":
    # Create an LSTM cell instance
    lstm_cell = LSTMCell(input_size=10, hidden_size=20)

    # Sample input: batch_size=5, input_size=10
    x = torch.randn(5, 10)

    # Without initial hidden state
    print('x :',x)
    h, c = lstm_cell(x)
    print("Hidden state shape:", h.shape)  # Expected: (5, 20)
    print("Cell state shape:", c.shape)  # Expected: (5, 20)

    # With initial hidden state
    h0 = torch.zeros(5, 20)
    c0 = torch.zeros(5, 20)
    h, c = lstm_cell(x, (h0, c0))
    print("Hidden state shape:", h.shape)  # Expected: (5, 20)
    print("Cell state shape:", c.shape)  # Expected: (5, 20)























