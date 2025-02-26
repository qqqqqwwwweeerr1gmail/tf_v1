import torch
import torch.nn as nn
import torch.optim as optim
import random

class InternalMotivationAgent(nn.Module):
    def __init__(self, num_neurons=3, output_range=10):
        super(InternalMotivationAgent, self).__init__()
        self.fc = nn.Linear(1, num_neurons)  # Input desired output, output 3 neuron activations
        self.output_range = output_range

    def forward(self, desired_output):
        neuron_activations = torch.sigmoid(self.fc(desired_output.float().unsqueeze(0)))
        return neuron_activations

    def apply_fixed_operator(self, neuron_activations):
        """
        Simulates the fixed operator (hands) that converts neuron activations to an integer output.
        This is a simplified example; a real operator might be more complex.
        """
        # Example: Weighted sum and rounding
        weights = torch.tensor([0.3, 0.5, 0.2])  # Fixed weights
        weighted_sum = torch.sum(neuron_activations * weights)
        output = torch.round(weighted_sum * (self.output_range - 1))
        return torch.clamp(output.long(), 0, self.output_range - 1)

def train_agent(agent, epochs=10000, learning_rate=0.01):
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Use MSE loss for continuous neuron activations

    for epoch in range(epochs):
        desired_output = torch.tensor(random.randint(0, agent.output_range - 1)) #randomly generate desired output
        neuron_activations = agent(desired_output)
        actual_output = agent.apply_fixed_operator(neuron_activations)

        # Intrinsic reward: closeness to desired output
        reward = -torch.abs(actual_output - desired_output).float()

        # Optimize based on the reward (which is derived from the MSE of neuron activations)
        optimizer.zero_grad()
        loss = criterion(neuron_activations, desired_output.float() / (agent.output_range-1)) #try to make the neuron activation close to the normalized desired output.
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Desired: {desired_output.item()}, Actual: {actual_output.item()}, Reward: {reward.item()}, Loss:{loss.item()}")

if __name__ == "__main__":
    agent = InternalMotivationAgent()

    # Test the trained agent
    for _ in range(5):
        test_desired = torch.tensor(random.randint(0, agent.output_range - 1))
        test_activations = agent(test_desired)
        test_actual = agent.apply_fixed_operator(test_activations)
        print(f"Test: Desired {test_desired.item()}, Actual {test_actual.item()}")


    train_agent(agent)

    # Test the trained agent
    for _ in range(5):
        test_desired = torch.tensor(random.randint(0, agent.output_range - 1))
        test_activations = agent(test_desired)
        test_actual = agent.apply_fixed_operator(test_activations)
        print(f"Test: Desired {test_desired.item()}, Actual {test_actual.item()}")























