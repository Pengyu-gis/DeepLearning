import matplotlib.pyplot as plt
import networkx as nx

# Define the network structure (input, hidden, output sizes)
input_size = 10
hidden_size = 5
output_size = 1

# Create a directed graph using networkx
G = nx.DiGraph()

# Add nodes for input layer
input_nodes = [f'I{i+1}' for i in range(input_size)]
G.add_nodes_from(input_nodes)

# Add nodes for hidden layer
hidden_nodes = [f'H{i+1}' for i in range(hidden_size)]
G.add_nodes_from(hidden_nodes)

# Add nodes for output layer
output_nodes = [f'O{i+1}' for i in range(output_size)]
G.add_nodes_from(output_nodes)

# Add edges between input and hidden layer
for i in input_nodes:
    for h in hidden_nodes:
        G.add_edge(i, h)

# Add edges between hidden and output layer
for h in hidden_nodes:
    for o in output_nodes:
        G.add_edge(h, o)

# Create layout for visualization
pos = {}
layer_dist = 1
# Input layer
for i, node in enumerate(input_nodes):
    pos[node] = (0, i)

# Hidden layer
for i, node in enumerate(hidden_nodes):
    pos[node] = (layer_dist, i)

# Output layer
for i, node in enumerate(output_nodes):
    pos[node] = (2 * layer_dist, i)

# Draw the network
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold', arrows=True)
plt.title('Visualization of Feed-Forward Neural Network', size=15)
plt.show()
