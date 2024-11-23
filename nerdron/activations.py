import numpy as np

class Activation:
    """Base class for activation functions."""
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class ReLU(Activation):
    """Rectified Linear Unit activation function."""
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def backward(self, grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        return grad * (x > 0)

class Sigmoid(Activation):
    """Sigmoid activation function."""
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def backward(self, grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return grad * s * (1 - s)

class Tanh(Activation):
    """Hyperbolic tangent activation function."""
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def backward(self, grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        return grad * (1 - np.tanh(x) ** 2)