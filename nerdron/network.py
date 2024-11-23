import numpy as np
from typing import List, Union, Optional
from .layers import Layer
from .metrics import MSE
from .utils import GPU

class Net:
    def __init__(self):
        self.layers: List[Layer] = []
        self.loss_function = None
        self.learning_rate = None
        self.compiled = False
        self.gpu = GPU()
    
    def add(self, layer: Layer) -> None:
        """Add a layer to the network."""
        if self.layers and not hasattr(layer, 'input_size'):
            layer.input_size = self.layers[-1].output_size
        self.layers.append(layer)
        layer.initialize()
    
    def compile(self, learning_rate: float = 0.01, loss: str = 'mse') -> None:
        """Compile the network with specified learning rate and loss function."""
        self.learning_rate = learning_rate
        self.loss_function = MSE() if loss.lower() == 'mse' else loss
        self.compiled = True
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation through the network."""
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Backward propagation through the network."""
        grad = self.loss_function.derivative(y_true, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.learning_rate)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
            batch_size: Optional[int] = None, verbose: bool = True) -> List[float]:
        """Train the network on given data."""
        if not self.compiled:
            raise RuntimeError("Model must be compiled before training")
        
        X = self.gpu.array(X)
        y = self.gpu.array(y)
        losses = []
        
        for epoch in range(epochs):
            if batch_size:
                indices = np.random.permutation(len(X))
                for i in range(0, len(X), batch_size):
                    batch_idx = indices[i:i + batch_size]
                    X_batch = X[batch_idx]
                    y_batch = y[batch_idx]
                    
                    y_pred = self.forward(X_batch)
                    self.backward(y_batch, y_pred)
            else:
                y_pred = self.forward(X)
                self.backward(y, y_pred)
            
            current_loss = self.loss_function(y, self.forward(X))
            losses.append(float(self.gpu.to_cpu(current_loss)))
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input samples."""
        X = self.gpu.array(X)
        predictions = self.forward(X)
        return self.gpu.to_cpu(predictions)
    
    def summary(self) -> None:
        """Print a summary of the network architecture."""
        print("\nModel Summary:")
        print("-" * 70)
        print(f"{'Layer Type':<20} {'Output Shape':<20} {'Params':<15}")
        print("=" * 70)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            params = layer.get_params_count()
            total_params += params
            print(f"{layer.__class__.__name__:<20} {str(layer.output_shape):<20} {params:<15}")
        
        print("-" * 70)
        print(f"Total params: {total_params}")
        print("-" * 70)


class Sequential(Net):
    """Sequential model for simple layer stacking."""
    def __init__(self, layers: Optional[List[Layer]] = None):
        super().__init__()
        if layers:
            for layer in layers:
                self.add(layer)
