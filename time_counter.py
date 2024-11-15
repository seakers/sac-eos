import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation_fn=nn.ReLU):
        """
        A customizable neural network class.

        Args:
            input_size (int): Number of input features.
            hidden_layers (list of int): List where each element represents the number of neurons in a hidden layer.
            output_size (int): Number of output features (e.g., classes for classification).
            activation_fn (torch.nn.Module): Activation function to use (default: ReLU).
        """
        super(NeuralNetwork, self).__init__()

        # Initialize layers
        layers = []

        # Input layer to the first hidden layer
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(activation_fn())

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(activation_fn())

        # Last hidden layer to output
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        # Combine layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Defines the forward pass of the network."""
        return self.network(x)

# Example usage
if __name__ == "__main__":
    # Create an instance of the network
    input_size = 100 * 20  # Number of input features
    hidden_layers = [1024, 64, 1024, 64]  # Two hidden layers with 64 and 32 neurons
    output_size = 1  # Single output (e.g., regression)

    model = NeuralNetwork(input_size, hidden_layers, output_size)

    # Print the network architecture
    print(model)

    # Example input
    x = torch.randn(5, input_size)  # Batch of 5 samples, each with 10 features

    # Forward pass
    import time

    start_time = time.perf_counter()

    output = model(x)

    # Stop the timer
    end_time = time.perf_counter()

    # Elapsed time in seconds
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")