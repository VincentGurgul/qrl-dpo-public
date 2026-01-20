'''Base module for vanilla neural networks.

Usage:
    This module is intended to be used in a supervised ML pipeline or to be
    subclassed, e.g. by the actor in a reinforcement learning context.
'''
import torch

from torch import nn

from utilities.model_training import set_seeds


class NeuralNetwork(nn.Module):
    '''Base class for a vanilla neural network / multilayer perceptron.

    Args:
        input_size (int): Size of the input layer.
        hidden_sizes (list of int): List of sizes for the hidden layers.
        output_size (int): Size of the output layer.
        hidden_activation: Activation function for the hidden layers. Can be
            any function R -> R. Defaults to ReLU.
        output_activation: Activation function for the output layer. Can be any
            function R -> R. Defaults to None, i.e. linear activation.
        seed (int): Random seed for reproducibility. Defaults to None, i.e.
            non-deterministic behaviour.
    '''
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        hidden_activation: callable = torch.relu,
        output_activation: callable = None,
        seed: int = None,
    ):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        if seed:
            set_seeds(seed)

        self.layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass through the network.'''

        for layer in self.layers:
            x = self.hidden_activation(layer(x))
        if self.output_activation:
            return self.output_activation(self.output_layer(x))
        else:
            return self.output_layer(x)
