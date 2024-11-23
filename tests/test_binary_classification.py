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