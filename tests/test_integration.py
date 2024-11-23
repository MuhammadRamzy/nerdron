import numpy as np
import pytest
from nerdron import Sequential, Dense, ReLU, Sigmoid, Tanh

def test_model_save_load(tmp_path):
    """Test model serialization and deserialization."""
    # Create and train a model
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, (100, 1))
    
    model = Sequential([
        Dense(4, input_size=2, activation=ReLU()),
        Dense(1, activation=Sigmoid())
    ])
    model.compile(learning_rate=0.01, loss='mse')
    model.fit(X, y, epochs=10, verbose=False)
    
    # Get predictions before saving
    original_predictions = model.predict(X)
    
    # Save weights to files
    weights_path = tmp_path / "weights.npz"
    np.savez(
        weights_path,
        *[layer.weights for layer in model.layers],
        *[layer.biases for layer in model.layers]
    )
    
    # Create new model with same architecture
    new_model = Sequential([
        Dense(4, input_size=2, activation=ReLU()),
        Dense(1, activation=Sigmoid())
    ])
    new_model.compile(learning_rate=0.01, loss='mse')
    
    # Load weights
    weights_dict = np.load(weights_path)
    for i, layer in enumerate(new_model.layers):
        layer.weights = weights_dict[f'arr_{i*2}']
        layer.biases = weights_dict[f'arr_{i*2+1}']
    
    # Get predictions after loading
    new_predictions = new_model.predict(X)
    
    # Assert predictions are the same
    np.testing.assert_array_almost_equal(original_predictions, new_predictions)

def test_complex_architecture():
    """Test a more complex model architecture."""
    model = Sequential([
        Dense(8, input_size=4, activation=ReLU()),
        Dense(6, activation=Tanh()),
        Dense(4, activation=ReLU()),
        Dense(1, activation=Sigmoid())
    ])
    
    # Test forward pass
    X = np.random.randn(10, 4)
    model.compile(learning_rate=0.01, loss='mse')
    output = model.predict(X)
    
    assert output.shape == (10, 1)
    assert np.all((output >= 0) & (output <= 1))

def test_batch_size_variations():
    """Test different batch sizes during training."""
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, (100, 1))
    
    model = Sequential([
        Dense(4, input_size=2, activation=ReLU()),
        Dense(1, activation=Sigmoid())
    ])
    model.compile(learning_rate=0.01, loss='mse')
    
    # Test different batch sizes
    batch_sizes = [1, 10, 32, 100]  # Including full batch
    for batch_size in batch_sizes:
        history = model.fit(X, y, epochs=5, batch_size=batch_size, verbose=False)
        assert len(history) == 5