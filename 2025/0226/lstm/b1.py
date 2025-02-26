import torch
import torch.nn as nn

class SingleLSTMCellModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SingleLSTMCellModel, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, input_seq, hidden_state=None):
        """
        Processes an input sequence using a single LSTM cell.

        Args:
            input_seq (torch.Tensor): Input sequence of shape (seq_len, batch_size, input_size).
            hidden_state (tuple, optional): Initial hidden state (h_0, c_0).
                                             If None, initializes with zeros.

        Returns:
            torch.Tensor: Output sequence of shape (seq_len, batch_size, hidden_size).
            tuple: Final hidden state (h_n, c_n).
        """
        seq_len, batch_size, _ = input_seq.size()
        outputs = torch.zeros(seq_len, batch_size, self.hidden_size)

        if hidden_state is None:
            h_t = torch.zeros(batch_size, self.hidden_size)
            c_t = torch.zeros(batch_size, self.hidden_size)
        else:
            h_t, c_t = hidden_state

        for t in range(seq_len):
            h_t, c_t = self.lstm_cell(input_seq[t], (h_t, c_t))
            outputs[t] = h_t

        return outputs, (h_t, c_t)

# Example usage:
input_size = 10
hidden_size = 20
seq_len = 5
batch_size = 3

model = SingleLSTMCellModel(input_size, hidden_size)
input_seq = torch.randn(seq_len, batch_size, input_size)

outputs, final_hidden_state = model(input_seq)

print("Output sequence shape:", outputs.shape)
print("Final hidden state h shape:", final_hidden_state[0].shape)
print("Final hidden state c shape:", final_hidden_state[1].shape)

# Example of providing an initial hidden state.
initial_h = torch.randn(batch_size, hidden_size)
initial_c = torch.randn(batch_size, hidden_size)
initial_hidden_state = (initial_h, initial_c)

outputs2, final_hidden_state2 = model(input_seq, initial_hidden_state)

print("Output sequence shape with initial hidden state:", outputs2.shape)

test_input = torch.Tensor([1,1,1])
outputs, (h_t, c_t) = model.forward(test_input)
print(outputs)
print(h_t)
print(c_t)



















