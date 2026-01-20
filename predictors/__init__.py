'''Base module for predictors.

Classes:
    NeuralNetwork: Vanilla neural network.
    QuantumNeuralNetwork: Variational quantum circuit based quantum neural
    network (QNN).

Usage:
    The classes are intended to be used in a supervised ML pipeline or to be
    subclassed by concrete predictor implementations, e.g. the actor in a 
    reinforcement learning context.
'''
from predictors.neural_network import NeuralNetwork
from predictors.quantum_neural_network import QuantumNeuralNetwork

__all__ = ['NeuralNetwork', 'QuantumNeuralNetwork']
