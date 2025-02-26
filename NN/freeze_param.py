import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # 2 input features, 2 hidden neurons
        self.fc2 = nn.Linear(2, 1)  # 2 hidden neurons, 1 output

    def forward(self, x):
        x = self.fc1(x)  # Linear activation y=x
        x = self.fc2(x)
        return x

# Initialize the network
net = SimpleNet()

# Print initial parameters
print("Initial Parameters:")
for name, param in net.named_parameters():
    print(name, param.data)

# Training data
inputs = torch.tensor([[1.0, 3.0], [4.0, 6.0]])
targets = torch.tensor([[4.0], [10.0]])

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Train the network
for epoch in range(100):  # Train for 100 epochs
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# Print parameters after training
print("\nParameters After Training:")
for name, param in net.named_parameters():
    print(name, param.data)


# Detach the variable and set requires_grad to False
net.fc1.weight = net.fc1.weight.detach()
# net.fc1.weight.requires_grad = False
# Freeze one hidden neuron (e.g., the first neuron in fc1)
net.fc1.weight[0].requires_grad = False
net.fc1.bias[0].requires_grad = False

# Train again with one hidden neuron frozen
for epoch in range(100):  # Train for another 100 epochs
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# Print parameters after freezing one hidden neuron
print("\nParameters After Freezing One Hidden Neuron and Training:")
for name, param in net.named_parameters():
    print(name, param.data)























