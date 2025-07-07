import torch
from typing import Union

class SimpleNet(torch.nn.Module):
    """
    Simple feedforward neural network with a variable number of layers and nodes per layer.

    Args:
        input_size (int): Number of input features
        architecture (list[int]): List of integers representing the number of nodes per layer
        use_batchnorm (bool): Whether to use batch normalization after each layer

    Returns:
        torch.nn.Module: A simple feedforward neural network
    """
    def __init__(
            self,
            input_size:int,
            architecture:Union[list[int],tuple[int]],
            use_batchnorm:bool=True,
            ):
        super(SimpleNet, self).__init__()
        self.input_size = input_size
        self.architecture = architecture
        self.use_batchnorm = use_batchnorm
        self.layers = torch.nn.ModuleList()

        self.layers.append(torch.nn.Linear(self.input_size, self.architecture[0]))
        if self.use_batchnorm:
            self.layers.append(torch.nn.BatchNorm1d(self.architecture[0]))

        for n_in, n_out in zip(self.architecture[:-1], self.architecture[1:]):
            self.layers.append(torch.nn.Linear(n_in, n_out))
            if self.use_batchnorm:
                self.layers.append(torch.nn.BatchNorm1d(n_out))

        self.layers.append(torch.nn.Linear(self.architecture[-1], 1))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.relu(x)
        
        x = self.layers[-1](x)

        return torch.sigmoid(x)


class ABCDiscoNet(torch.nn.Module):
    """
    ABCDiscoNet model with one or more neural networks.

    Args:
        networks (list[torch.nn.Module]): List of neural networks

    Returns:
        torch.nn.Module: An ABCDiscoNet model
    """
    def __init__(
            self,
            networks: list[torch.nn.Module],
        ):
        super(ABCDiscoNet, self).__init__()
        self.dnn = torch.nn.ModuleList(networks)
    
    def __len__(self):
        return len(self.dnn)

    def forward(self, x):
        return [dnn(x) for dnn in self.dnn]