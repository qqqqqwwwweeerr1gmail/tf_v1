import numpy as np


# Simulate the brain-hand system
class BrainHandSystem:
    def __init__(self):
        # Initial state (last output) starts as zero vector
        self.last_output = np.array([0.0, 0.0, 0.0])

    def brain_process(self, will_vector):
        # Simplified: Map 4D will vector to 3D command (e.g., drop last dimension)
        brain_output = np.array([will_vector[0], will_vector[1], will_vector[2]])
        return brain_output

    def compute_output(self, will_vector):
        # Print the input state (last output) before processing
        print(f"Input state (last output): {self.last_output}")

        # Brain generates its output from the 4D will vector
        brain_output = self.brain_process(will_vector)
        print(f"Brain output: {brain_output}")

        # New output is brain output plus last output state
        new_output = brain_output + self.last_output
        print(f"New output (brain output + last state): {new_output}")

        # Update the last output state
        self.last_output = new_output
        return new_output


# Example usage
system = BrainHandSystem()
target_vector = np.array([5.0, 5.0, 5.0])  # Example target for the hand

# Simulate a few steps
for step in range(3):
    print(f"\nStep {step + 1}:")
    # Brain generates a "will" vector (here, just a sample; in reality, it could adapt)
    will_vector = np.array([1.0, 2.0, 3.0, 4.0])  # 4D input
    output = system.compute_output(will_vector)
    error = target_vector - output
    print(f"Error from target {target_vector}: {error}")























