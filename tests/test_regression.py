# File: tests/test_binary_classification.py
import numpy as np
import pytest
from nerdron import Sequential, Dense, ReLU, Sigmoid
from nerdron.utils import GPU

def test_xor_problem():
    """Test the network on the XOR problem."""
    # Prepare data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create model
    model = Sequential([
        Dense(4, input_size=2, activation=ReLU()),
        Dense(1, activation=Sigmoid())
    ])
    
    # Compile and train
    model.compile(learning_rate=0.01, loss='mse')
    history = model.fit(X, y, epochs=1000, verbose=False)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Assertions
    assert len(history) == 1000
    assert history[-1] < 0.1  # Final loss should be small
    assert np.all(np.abs(predictions - y) < 0.2)  # Predictions should be close to targets

def test_binary_classification_shapes():
    """Test shape handling in binary classification."""
    model = Sequential([
        Dense(4, input_size=2, activation=ReLU()),
        Dense(1, activation=Sigmoid())
    ])
    
    # Test incorrect input shapes
    with pytest.raises(ValueError):
        X_wrong = np.array([[0, 0, 0]])  # Wrong input dimension
        model.predict(X_wrong)
    
    # Test correct shapes
    X_correct = np.array([[0, 0]])
    output = model.predict(X_correct)
    assert output.shape == (1, 1)

# File: tests/test_regression.py
import numpy as np
import pytest
from nerdron import Sequential, Dense, ReLU, Tanh
from nerdron.metrics import MSE

def generate_regression_data(n_samples=100):
    """Generate synthetic regression data."""
    X = np.random.uniform(-10, 10, (n_samples, 1))
    y = 0.5 * X + 2 + np.random.normal(0, 0.1, (n_samples, 1))
    return X, y

def test_linear_regression():
    """Test the network on a simple regression problem."""
    # Generate data
    X, y = generate_regression_data()
    
    # Create model
    model = Sequential([
        Dense(8, input_size=1, activation=ReLU()),
        Dense(1, activation=None)  # Linear output for regression
    ])
    
    # Compile and train
    model.compile(learning_rate=0.001, loss='mse')
    history = model.fit(X, y, epochs=500, verbose=False)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Assertions
    assert len(history) == 500
    assert history[-1] < history[0]  # Loss should decrease
    
    # Test prediction shape
    assert predictions.shape == y.shape
    
    # Test reasonable predictions
    mse = MSE()(y, predictions)
    assert mse < 1.0  # Reasonable error threshold

def test_regression_batch_training():
    """Test batch training for regression."""
    X, y = generate_regression_data(1000)
    
    model = Sequential([
        Dense(8, input_size=1, activation=ReLU()),
        Dense(1)
    ])
    
    model.compile(learning_rate=0.001, loss='mse')
    history = model.fit(X, y, epochs=100, batch_size=32, verbose=False)
    
    assert len(history) == 100
    assert history[-1] < history[0]