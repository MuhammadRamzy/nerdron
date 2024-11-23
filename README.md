#  Nerdron: Your AI Sidekick

Nerdron is a lightweight, educational neural network library designed for simplicity and flexibility. Perfect for learning deep learning concepts or quickly prototyping custom neural networks.

## 🌟 Features

- **Simple API**: Build networks with just a few lines of code
- **GPU Support**: Automatic GPU acceleration with CuPy
- **Modular Design**: Easily extend with custom layers and activations
- **Educational**: Clear, documented code for learning deep learning concepts
- **Flexible Architecture**: Build any network architecture you need
- **Built-in Optimizations**: Batch training, GPU support, and more

## 🚀 Quick Start

### Installation

```bash
# From PyPI
pip install nerdron

# From source
git clone https://github.com/MuhammadRamzy/nerdron
cd nerdron
pip install -e .
```

### Simple Example: XOR Problem

```python
from nerdron import Sequential, Dense, ReLU, Sigmoid
import numpy as np

# Create data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create model
model = Sequential([
    Dense(4, input_size=2, activation=ReLU()),
    Dense(1, activation=Sigmoid())
])

# Compile and train
model.compile(learning_rate=0.01, loss='mse')
model.fit(X, y, epochs=1000)

# Make predictions
predictions = model.predict(X)
print("Predictions:", predictions)
```

## 📖 Detailed Documentation

### Creating a Model

#### Sequential Model

```python
from nerdron import Sequential, Dense, ReLU, Sigmoid

# Method 1: Pass layers list
model = Sequential([
    Dense(4, input_size=2, activation=ReLU()),
    Dense(1, activation=Sigmoid())
])

# Method 2: Add layers one by one
model = Sequential()
model.add(Dense(4, input_size=2, activation=ReLU()))
model.add(Dense(1, activation=Sigmoid()))
```

#### Available Layers

- **Dense**: Fully connected layer
  ```python
  Dense(units, input_size=None, activation=None)
  ```

#### Activation Functions

- ReLU: `ReLU()`
- Sigmoid: `Sigmoid()`
- Tanh: `Tanh()`

### Training

```python
# Compile model
model.compile(learning_rate=0.01, loss='mse')

# Train with full batch
model.fit(X, y, epochs=1000)

# Train with mini-batches
model.fit(X, y, epochs=1000, batch_size=32)

# Train with verbose output
model.fit(X, y, epochs=1000, verbose=True)
```

### Model Summary

```python
model.summary()
```

Output example:
```
Model Summary:
----------------------------------------------------------------------
Layer Type           Output Shape         Params     
======================================================================
Dense               (None, 4)            12         
Dense               (None, 1)            5          
----------------------------------------------------------------------
Total params: 17
----------------------------------------------------------------------
```

## 🧪 Running Tests

### Setting Up Test Environment

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_binary_classification.py

# Run tests with coverage report
pytest --cov=nerdron tests/
```

### Test Files Structure

```
tests/
├── conftest.py                    # Test configurations and fixtures
├── test_binary_classification.py  # Binary classification tests
├── test_regression.py            # Regression tests
├── test_components.py            # Individual component tests
└── test_integration.py           # Integration tests
```

## 🔧 Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nerdron.git
cd nerdron
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Run tests to verify setup:
```bash
pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Examples

### Regression Example

```python
from nerdron import Sequential, Dense, ReLU
import numpy as np

# Generate synthetic data
X = np.random.uniform(-10, 10, (1000, 1))
y = 0.5 * X + 2 + np.random.normal(0, 0.1, (1000, 1))

# Create model
model = Sequential([
    Dense(8, input_size=1, activation=ReLU()),
    Dense(1)  # No activation for regression
])

# Train
model.compile(learning_rate=0.001, loss='mse')
model.fit(X, y, epochs=500, batch_size=32)

# Predict
predictions = model.predict(X)
```

### Multi-Layer Network

```python
from nerdron import Sequential, Dense, ReLU, Tanh, Sigmoid

model = Sequential([
    Dense(16, input_size=4, activation=ReLU()),
    Dense(8, activation=Tanh()),
    Dense(4, activation=ReLU()),
    Dense(1, activation=Sigmoid())
])
```

## 🎯 Roadmap

- [ ] Additional optimization algorithms (Adam, RMSprop)
- [ ] Convolutional layers
- [ ] Dropout layer
- [ ] Model saving/loading
- [ ] Learning rate scheduling
- [ ] Early stopping
- [ ] More loss functions
- [ ] Batch normalization

## 📊 Performance Tips

1. **GPU Acceleration**
   - Install CuPy for automatic GPU support
   - Larger batch sizes work better on GPU

2. **Memory Management**
   - Use appropriate batch sizes for your memory constraints
   - Clear model outputs when not needed

3. **Training Optimization**
   - Start with a smaller learning rate
   - Increase epochs for complex problems
   - Monitor loss to avoid overfitting

## 🤔 Common Issues

1. **Installation Issues**
   ```bash
   # If CuPy installation fails
   pip install cupy-cuda11x  # Replace with your CUDA version
   ```

2. **Memory Errors**
   - Reduce batch size
   - Use CPU for smaller datasets

3. **Poor Convergence**
   - Adjust learning rate
   - Increase network capacity
   - Check data preprocessing

## 🙏 Acknowledgments

- NumPy team for the excellent array operations library
- CuPy team for GPU acceleration support
- Deep learning community for inspiration and algorithms

## 📫 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/nerdron](https://github.com/yourusername/nerdron)