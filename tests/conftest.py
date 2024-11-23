import pytest
import numpy as np

@pytest.fixture
def sample_data():
    """Fixture to provide sample data for tests."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    return X, y

@pytest.fixture
def simple_model():
    """Fixture to provide a simple compiled model."""
    from nerdron import Sequential, Dense, ReLU, Sigmoid
    
    model = Sequential([
        Dense(4, input_size=2, activation=ReLU()),
        Dense(1, activation=Sigmoid())
    ])
    model.compile(learning_rate=0.01, loss='mse')
    return model