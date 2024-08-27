# 3. MLP Class
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU()])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)



'''
 constructor for the MLP class. It takes three parameters:

input_size: The number of input features
hidden_sizes: A list of integers representing the number of neurons in each hidden layer
output_size: The number of output neurons

prev_size = input_size
This initializes a variable to keep track of the input size for each layer. It starts with the initial input size.

layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU()])

For each hidden layer:
It adds a linear (fully connected) layer from the previous size to the current hidden size
It adds a ReLU activation function after the linear layer
extend() is used to add multiple items to the list at once

prev_size = hidden_size
This updates the prev_size for the next iteration, so the next layer knows its input size.

layers.append(nn.Linear(prev_size, output_size))

After the loop, this adds a final linear layer that outputs to the specified output size.
self.layers = nn.Sequential(*layers)
This creates a Sequential container with all the layers we've added:

nn.Sequential is a container that runs layers in sequence
*layers unpacks the list of layers into individual arguments

The asterisk * is the unpacking operator in Python
'''
