import torch
import torch.nn as nn




lstm_cell = nn.LSTMCell(1, 1)
print(lstm_cell)
tensor = torch.randn(1, 1, 2)
out = lstm_cell.forward(tensor)
print(out)




























