import torch
import torch.nn as nn
import torch.optim as optim

# Define the feed-forward neural network class
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        
        # Define the layers: an input layer, one hidden layer, and an output layer
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer 1
        self.relu = nn.ReLU()  # Activation function (ReLU)
        self.fc2 = nn.Linear(hidden_size, output_size)  # Fully connected layer 2
        
    def forward(self, x):
        out = self.fc1(x)  # Input to hidden
        out = self.relu(out)  # Apply ReLU activation
        out = self.fc2(out)  # Hidden to output
        return out

# Example usage:

# Define the network's parameters
input_size = 10   # Number of input features
hidden_size = 5   # Number of neurons in the hidden layer
output_size = 1   # Number of output features (e.g., for regression)

# Initialize the network
model = FeedForwardNN(input_size, hidden_size, output_size)

# Example input: a batch of 3 samples, each with 10 features
x = torch.randn(3, input_size)

# Forward pass through the network
output = model(x)

# Define a simple loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss (for regression tasks)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example target (ground truth) for the batch
target = torch.randn(3, output_size)

# Training loop (just a single step for demonstration)
optimizer.zero_grad()  # Zero the gradients
loss = criterion(output, target)  # Compute the loss
loss.backward()  # Backpropagate the loss
optimizer.step()  # Update the network's weights

print(f"Output: {output}")
print(f"Loss: {loss.item()}")
