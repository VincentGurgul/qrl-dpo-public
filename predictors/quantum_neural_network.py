'''Base module for quantum neural networks.

Usage:
    This module is intended to be used in a supervised ML pipeline or to be
    subclassed, e.g. by the actor in a reinforcement learning context.
'''
import re
import warnings
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pennylane as qml

from pennylane.operation import Operation
from pennylane.ops import StatePrep

from custom_warnings import ConvergenceWarning, PerformanceWarning

from . import input_transformations

from utilities.model_training import set_seeds


class AngleEncoding(Operation):
    '''Standard Angle Embedding without entanglement layers.

    Args:
        features (tensor_like): Input tensor of shape '(N,)', where N is the
            number of input features to embed, with N <= n.
        wires (Any or Iterable[Any]): Wires that the template acts on.
        rotation (qml.operation.Operator): PennyLane rotation gate class.
        id (str): Custom label given to an operator instance, useful for
            identifying specific instances.

    Raises:
        ValueError if number of features exceeds number of qubits.
    '''
    def __init__(self, features, wires, rotation, id=None):
        super().__init__(features, wires=wires, id=id)
        self._hyperparameters = {'rotation': rotation}

        shape = qml.math.shape(features)[-1:]
        n_features = shape[0]
        if n_features > len(wires):
            raise ValueError(f'Angle encoding selected, but number of features ({n_features}) exceeds number of qubits ({len(wires)}).')

    @staticmethod
    def compute_decomposition(features, wires, rotation):  # pylint: disable=arguments-differ
        '''Representation of the operator as a product of other operators.

        Args:
            features (tensor_like): Input tensor of dimension '(len(wires),)'.
            wires (Any or Iterable[Any]): Wires that the operator acts on.
            rotation (qml.operation.Operator): PennyLane rotation gate class.

        Returns:
            list[.Operator]: Decomposition of the operator.
        '''
        batched = qml.math.ndim(features) > 1
        features = qml.math.T(features) if batched else features
        return [rotation(features[i], wires=wires[i]) for i in range(len(wires))]
    

class StackedAngleEncoding(Operation):
    '''Stacked Angle Embedding with entanglement layers after encoding all
    qubits.

    Args:
        features (tensor_like): Input tensor of shape '(N,)', where N is the
            number of input features to embed.
        wires (Any or Iterable[Any]): Wires that the template acts on.
        rotation (qml.operation.Operator): PennyLane rotation gate class.
        entanglement (str): Type of entanglement ('full', 'linear', or 'reverse_linear').
    '''
    def __init__(self, features, wires, rotation, entanglement):
        super().__init__(features, wires=wires, id=None)
        self._hyperparameters = {'rotation': rotation, 'entanglement': entanglement}

    @staticmethod
    def compute_decomposition(features, wires, rotation, entanglement):
        '''Representation of the operator as a product of other operators.

        Args:
            features (tensor_like): Input tensor of dimension '(len(wires),)'.
            wires (Any or Iterable[Any]): Wires that the operator acts on.
            rotation (qml.operation.Operator): PennyLane rotation gate class.
            entanglement (str): Type of entanglement ('full', 'linear', or 'reverse_linear').

        Returns:
            list[.Operator]: Decomposition of the operator.
        '''
        operations = []
        num_qubits = len(wires)
        num_features = qml.math.shape(features)[-1]
        batched = qml.math.ndim(features) > 1

        # Iterate through the features in a stacked manner
        for i in range(num_features):
            target_wire = wires[i % num_qubits]
            feature_value = features[..., i] if batched else features[i]
            operations.append(rotation(feature_value, wires=target_wire))

            # Add entanglement layer after each complete pass through all qubits
            if (i + 1) % num_qubits == 0 and entanglement:
                if entanglement == 'linear':
                    for j in range(num_qubits - 1):
                        operations.append(qml.CNOT(wires=[j, j + 1]))
                elif entanglement == 'reverse_linear':
                    for j in range(num_qubits - 1, 0, -1):
                        operations.append(qml.CNOT(wires=[j - 1, j]))
                elif entanglement == 'full':
                    for j in range(num_qubits):
                        if i != j:
                            operations.append(qml.CNOT(wires=[i, j]))

        return operations
    

class AmplitudeEncoding(StatePrep):
    '''Encodes a feature vector into the amplitude vector of an n-qubit quantum
    state.

    Args:
        features (tensor_like): Input tensor of dimension `(2^len(wires),)` or
            less if `pad_with` is specified.
        wires (Any or Iterable[Any]): Wires that the template acts on.
        pad_with (float or complex): Constant to pad the input to size `2^n`,
            if needed.
        normalize (bool): Whether to automatically normalize the input features.
        id (str): Custom label for identifying the operator instance.
        validate_norm (bool): Whether to validate the norm of the input state.

    Raises:
        ValueError if number of features exceeds Hilbert dimension.
    '''
    def __init__(
            self,
            features,
            wires,
            pad_with=None,
            normalize=False,
            validate_norm=True,
    ):
        super().__init__(
            features,
            wires=wires,
            pad_with=pad_with,
            normalize=normalize,
            validate_norm=validate_norm,
            id=None,
        )
        shape = qml.math.shape(features)[-1:]
        n_features = shape[0]
        hilbert_dim = 2**len(wires)
        if n_features > hilbert_dim:
            raise ValueError(f'Amplitude encoding selected, but number of features ({n_features}) exceeds Hilbert dimension of 2^num_qubits ({hilbert_dim}).')


class ParameterizedQuantumCircuit:
    '''Generates a parameterized quantum circuit with specified rotations and
    entanglement patterns that can be used to serve as an ansatz for quantum
    machine learning models. The circuit is defined by rotation gates applied to
    individual qubits and optional entanglement gates between qubits.

    Args:
        num_qubits (int): Disired number of qubits in the quantum circuit.
        rotation_axes (str, optional): Bloch sphere axes to rotate around in the
            parameterized quantum circuit. Can be any combination of 'x', 'y'
            and 'z', e.g. 'xy' or 'xyz'. If multiple letters are passed,
            multiple rotations will happen. Defaults to 'y'.
        entanglement (str, optional): Type of entanglement pattern ('linear',
        'reverse_linear', or 'full'). Defaults to 'reverse_linear'.

    Methods:
        __call__: Constructs the parameterized quantum circuit using the provided weights.
    '''
    def __init__(
        self,
        num_qubits: int,
        rotation_axes: str = 'y',
        entanglement: str = 'reverse_linear',
    ):
        self.num_qubits = num_qubits
        self.rotation_axes = rotation_axes
        self.entanglement = entanglement

    def __call__(self, weights: torch.Tensor):
        '''Applies parameterized rotation gates (RX, RY, RZ) to all qubits
        based on the specified axes, followed by entanglement gates as defined
        by the entanglement pattern.

        Args:
            weights (torch.Tensor): A tensor containing the parameters for the
                rotation gates.
        '''
        for i, param in enumerate(weights.flatten()):

            # Apply rotations to all qubits
            if 'x' in self.rotation_axes:
                qml.RX(param, wires=i % self.num_qubits)
            if 'y' in self.rotation_axes:
                qml.RY(param, wires=i % self.num_qubits)
            if 'z' in self.rotation_axes:
                qml.RZ(param, wires=i % self.num_qubits)

            # After each full pass through all qubits, add entanglement
            if (i + 1) % self.num_qubits == 0 and (i + 1) != len(weights):
                if self.entanglement == 'linear':
                    for j in range(self.num_qubits - 1):
                        qml.CNOT(wires=[j, j + 1])
                elif self.entanglement == 'reverse_linear':
                    for j in range(self.num_qubits - 1, 0, -1):
                        qml.CNOT(wires=[j - 1, j])
                elif self.entanglement == 'full':
                    for i in range(self.num_qubits):
                        for j in range(self.num_qubits):
                            if i != j:
                                qml.CNOT(wires=[i, j])


class QuantumNeuralNetwork(nn.Module):
    '''Quantum Neural Network class using Pennylane and Pytorch.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        num_qubits (int): Number of qubits.
        num_qubits (int, optional): Number of qubits to use in the quantum
            circuit. Defaults to None, i.e. minimal number of qubits required
            to encode the provided input size is chosen.
        num_weights (int, optional): Number of weights for the circuit.
            Defaults to num_qubits.
        output_map (tuple, optional): Desired output mapping range. Defaults
            to `None`.
        rotation_axes (str, optional): Bloch sphere axes to rotate around in
            the parameterized quantum circuit. Can be any combination of 'x',
            'y' and 'z', e.g. 'xy' or 'xyz'. If multiple letters are passed,
            multiple rotations will happen. Defaults to 'y'.
        classical_layers (bool, optional): Whether to add classical FNN layers
            after the variational quantum circuit. Defaults to False.
        output_activation (callable, optional): Activation function to apply 
            to outputs. Defaults to `None`.
        encoding (str, optional): Input encoding strategy ('angle',
            'stacked_angle' or 'amplitude'). Defaults to 'amplitude'.
        input_transformation (callable, optional): Function to transform input with.
            Can be "standardize", "normalized_arcsin", "min_max_scale_to_range",
            "radial_to_linear", or None. Defaults to None.
        entanglement (str, optional): Type of entanglement structure to use in
            between rotation layers. Can be 'full', 'linear' or 'reverse_linear'.
            Defaults to 'reverse_linear'.
        device (str, optional): Device to run computations on. Can be any device
            supported by torch, e.g. 'cpu', 'gpu', 'mps'. Defaults to 'cpu'.
        seed (int, optional): Random seed for reproducibility. If not set, forward passes will be deterministic, but model initializations will be random. If set, the model is fully deterministic. Defaults to None, i.e. no seeding.

    Raises:
        ValueError: If the inputs, outputs, or encoding configurations are
            invalid.
    '''
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_qubits: int = None,
        num_weights: int = None,
        output_map: tuple = None,
        rotation_axes: str = 'y',
        classical_layers: bool = False,
        output_activation: callable = None,
        encoding: str = 'amplitude',
        input_transformation: callable = None,
        entanglement: str = 'reverse_linear',
        device: str = 'cpu',
        seed: int = None,
    ):
        # Catch erroneous model configurations
        if not bool(re.fullmatch(r'[xyz]*', rotation_axes)):
            raise ValueError(f'Rotation option "{rotation_axes}" not recognized. Please only use the letters "x", "y", or "z".')
        if 'x' not in rotation_axes and 'y' not in rotation_axes:
            ConvergenceWarning.warn('QNN does not converge with only Z rotations.')
        if entanglement not in ['linear', 'reverse_linear', 'full']:
            raise ValueError(f'Entanglement option {entanglement} not recognized. Please specify "linear", "reverse_linear", or "full".')
        if input_transformation is not None and input_transformation.__name__ not in dir(input_transformations):
            raise ValueError(f'Input transformation "{input_transformation.__name__}" not recognized. Please specify "standardize", "normalized_arcsin", "min_max_scale_to_range", "radial_to_linear", or None.')
        if encoding not in ['angle', 'stacked_angle', 'amplitude']:
            raise ValueError(f'Encoding option {encoding} not recognized. Please specify "angle" or "amplitude".')  
        if input_transformation in [input_transformations.radial_to_linear, input_transformations.radial_to_linear_small] and encoding == 'angle':
            PerformanceWarning.warn('Radial-to-linear input transformation is not recommended in combination with Angle Encoding due to slow computation, slow convergence and high likelihood of kernel crashing.')

        # Initialize instance and attributes
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_weights = num_weights
        self.output_map = output_map
        self.rotation_axes = rotation_axes
        self.classical_layers = classical_layers
        self.output_activation = output_activation
        self.encoding = encoding
        self.input_transformation = input_transformation
        self.entanglement = entanglement
        self.device = device

        # Set random seed for reproducibility if provided
        if seed:
            set_seeds(seed)

        # Initialize input size
        if input_transformation == input_transformations.radial_to_linear:
            self.actual_input_size = input_size * 4
        elif input_transformation == input_transformations.radial_to_linear_small:
            self.actual_input_size = input_size * 2
        else:
            self.actual_input_size = input_size

        # Initialize number of qubits
        # Uses minimal number of qubits required to encode the provided data
        if num_qubits is not None:
            self.num_qubits = num_qubits
        elif encoding == 'angle' or encoding == 'stacked_angle':
            self.num_qubits = np.max((self.actual_input_size, self.output_size))
        elif encoding == 'amplitude':
            self.num_qubits = np.max((int(np.ceil(np.log2(self.actual_input_size))), self.output_size))

        # Raise errors / warnings for excessive number of qubits
        if self.num_qubits > 64:
            raise ValueError('Number of qubits exceeds 64, the maximum dimension supported for NumPy ndarrays, which are used as the basis of PennyLane statevector simulation. Please reduce the number of qubits or run computations on QPU.')
        elif self.num_qubits > 20:
            PerformanceWarning.warn('Number of qubits exceeds 20, which may lead to slow computation times and potential kernel crashes. Consider reducing the number of qubits or running computations on QPU.')

        # Initialize number of weights to number of qubits if not specified
        if self.num_weights is None:
            self.num_weights = self.num_qubits

        # Initialize Ansatz
        self.ansatz = ParameterizedQuantumCircuit(
            self.num_qubits, rotation_axes, entanglement)

        # Define quantum circuit for Pennylane QNode
        def qnode(inputs, weights):

            # Encode inputs to a quantum state
            if encoding == 'angle':
                AngleEncoding(
                    features=inputs,
                    wires=range(self.num_qubits),
                    rotation=qml.RX if 'x' in rotation_axes else qml.RY,
                )
            elif encoding == 'stacked_angle':
                StackedAngleEncoding(
                    features=inputs,
                    wires=range(self.num_qubits),
                    rotation=qml.RX if 'x' in rotation_axes else qml.RY,
                    entanglement=entanglement,
                )
            elif encoding == 'amplitude':
                AmplitudeEncoding(
                    features=inputs,
                    wires=range(self.num_qubits),
                    normalize=True,
                    pad_with=0,
                )

            # Build Ansatz with the provided weights
            self.ansatz(weights)

            # Measure num_outputs first qubits Z amplitude
            if classical_layers:
                return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
            else:
                return [qml.expval(qml.PauliZ(i)) for i in range(self.output_size)]

        # Initialize the Pennylane QNode and TorchLayer
        self.quantum_circuit = qml.QNode(
            func=qnode,
            device=qml.device('default.qubit', wires=self.num_qubits, shots=None),
            interface='torch',
            diff_method='best',
        )
        self.qlayer = qml.qnn.TorchLayer(
            qnode=self.quantum_circuit,
            weight_shapes={'weights': (self.num_weights,)},
        )

    def __map_to_output_range(self, logits):
        '''
        Maps the logits in the interval [-1, 1] to the desired output range.

        Args:
            logits (list, np.ndarray, or tensor): Logits in the interval [-1, 1].

        Returns:
            same type as logits: Values mapped to the set output range.
        '''
        if self.output_map == (-1, 1):
            return logits
        elif self.output_map == (-float('inf'), float('inf')):
            # Apply hyperbolic tangent mapping for infinite range
            return 100 * torch.arctanh(torch.clamp(logits, -0.9999, 0.9999))
        else:
            # Scale logits to fit specified range
            normalized_output = (logits + 1) / 2
            return self.output_map[0] + normalized_output * (self.output_map[1] - self.output_map[0])

    def forward(self, tensor: torch.tensor) -> torch.tensor:
        '''
        Performs a forward pass through the quantum neural network.

        Args:
            input (torch.tensor): Input tensor of shape [batch_size, num_inputs].

        Returns:
            torch.tensor: Output tensor of shape [batch_size, num_outputs].
        '''
        if self.input_transformation is not None:
            tensor = self.input_transformation(tensor)
        if len(tensor) != self.actual_input_size and tensor.shape[-1] != self.actual_input_size:
            raise ValueError(f'Provided tensor has shape {tensor.shape}, but input size is configured to {self.actual_input_size}')
        if self.classical_layers:
            # tensor = torch.nn.Linear(self.num_inputs, self.num_qubits, dtype=torch.float64)(tensor)
            tensor = self.qlayer(tensor).to(self.device)
            tensor = torch.nn.Linear(self.num_qubits, self.output_size, dtype=torch.float64)(tensor)
        else:
            tensor = self.qlayer(tensor).to(self.device)
        if self.output_map:
            tensor = self.__map_to_output_range(tensor)
        if self.output_activation:
            tensor = self.output_activation(tensor)
        return tensor

    def draw(self):
        '''Draws the quantum circuit diagram as plain text using PennyLane's
        drawing functionality.
        '''
        with warnings.catch_warnings(action='ignore'):
            print(qml.draw(self.quantum_circuit, level='device')(
                np.zeros(self.actual_input_size),
                np.zeros(self.num_weights)))

    def draw_mpl(self):
        '''Draws the quantum circuit diagram in Matplotlib using PennyLane's
        MPL drawing functionality.
        '''
        with warnings.catch_warnings(action='ignore'):
            qml.draw_mpl(self.quantum_circuit, level='device')(
                np.zeros(self.actual_input_size),
                np.zeros(self.num_weights))
            plt.show()
