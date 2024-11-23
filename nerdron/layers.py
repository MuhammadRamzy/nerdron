import numpy as np
from typing import Optional, Tuple, Union
from .activations import ReLU

class Layer:
    """Base class for all layers."""
    def __init__(self):
        self.input_size = None
        self.output_size = None
        self.input = None
        self.output = None
    
    def initialize(self) -> None:
        """Initialize layer parameters."""
        pass
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass."""
        raise NotImplementedError
    
    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass."""
        raise NotImplementedError
    
    def get_params_count(self) -> int:
        """Return the number of trainable parameters."""
        return 0

class Dense(Layer):
    """Fully connected layer."""
    def __init__(self, units: int, input_size: Optional[int] = None,
                 activation: Optional[object] = None):
        super().__init__()
        self.units = units
        self.input_size = input_size
        self.activation = activation or ReLU()
        self.weights = None
        self.biases = None
        self.output_size = units
        self.output_shape = None
    
    def initialize(self) -> None:
        """Initialize weights and biases."""
        if self.input_size is None:
            raise ValueError("Input size must be specified for first layer")
        
        # He initialization
        scale = np.sqrt(2.0 / self.input_size)
        self.weights = np.random.randn(self.input_size, self.units) * scale
        self.biases = np.zeros((1, self.units))
        self.output_shape = (None, self.units)
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.activation.forward(self.output)
    
    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass through the layer."""
        grad = self.activation.backward(grad, self.output)
        
        # Compute gradients
        input_grad = np.dot(grad, self.weights.T)
        weights_grad = np.dot(self.input.T, grad)
        biases_grad = np.sum(grad, axis=0, keepdims=True)
        
        # Update parameters
        self.weights -= learning_rate * weights_grad
        self.biases -= learning_rate * biases_grad
        
        return input_grad
    
    def get_params_count(self) -> int:
        """Return the number of trainable parameters."""
        return self.weights.size + self.biases.size