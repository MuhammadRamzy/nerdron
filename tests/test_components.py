import numpy as np
import pytest
from nerdron import (
    Net, Sequential, Dense, ReLU, Sigmoid, Tanh, MSE, GPU
)

def test_activations():
    """Test activation functions."""
    x = np.array([-1.0, 0.0, 1.0])
    
    # Test ReLU
    relu = ReLU()
    relu_output = relu.forward(x)
    np.testing.assert_array_equal(relu_output, np.array([0.0, 0.0, 1.0]))
    
    # Test Sigmoid
    sigmoid = Sigmoid()
    sigmoid_output = sigmoid.forward(x)
    assert np.all((sigmoid_output >= 0) & (sigmoid_output <= 1))
    
    # Test Tanh
    tanh = Tanh()
    tanh_output = tanh.forward(x)
    assert np.all((tanh_output >= -1) & (tanh_output <= 1))

def test_dense_layer():
    """Test Dense layer functionality."""
    layer = Dense(3, input_size=2, activation=ReLU())
    layer.initialize()
    
    # Test shapes
    assert layer.weights.shape == (2, 3)
    assert layer.biases.shape == (1, 3)
    
    # Test forward pass
    x = np.array([[1.0, 2.0]])
    output = layer.forward(x)
    assert output.shape == (1, 3)
    
    # Test backward pass
    grad = np.ones((1, 3))
    input_grad = layer.backward(grad, 0.01)
    assert input_grad.shape == (1, 2)

def test_sequential_model():
    """Test Sequential model construction and methods."""
    model = Sequential([
        Dense(4, input_size=2, activation=ReLU()),
        Dense(3, activation=Sigmoid()),
        Dense(1, activation=None)
    ])
    
    # Test layer stacking
    assert len(model.layers) == 3
    assert model.layers[0].input_size == 2
    assert model.layers[0].output_size == 4
    assert model.layers[1].input_size == 4
    assert model.layers[1].output_size == 3
    assert model.layers[2].input_size == 3
    assert model.layers[2].output_size == 1

def test_mse_loss():
    """Test Mean Squared Error loss function."""
    mse = MSE()
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([[1.1], [2.1], [2.9]])
    
    # Test loss calculation
    loss = mse(y_true, y_pred)
    assert isinstance(loss, float)
    assert loss >= 0
    
    # Test gradient calculation
    grad = mse.derivative(y_true, y_pred)
    assert grad.shape == y_true.shape

def test_gpu_utils():
    """Test GPU utility functions."""
    gpu = GPU()
    
    # Test array conversion
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    gpu_x = gpu.array(x)
    
    # Test conversion back to CPU
    cpu_x = gpu.to_cpu(gpu_x)
    np.testing.assert_array_equal(x, cpu_x)

def test_model_compilation():
    """Test model compilation and validation."""
    model = Sequential()
    
    # Test compilation without layers
    with pytest.raises(ValueError):
        model.compile(learning_rate=0.01)
    
    # Test proper compilation
    model.add(Dense(2, input_size=2))
    model.compile(learning_rate=0.01, loss='mse')
    assert model.compiled
    assert model.learning_rate == 0.01
    assert isinstance(model.loss_function, MSE)