# Differentiable Programming

PyTorch-like automatic differentiation framework with tensor operations and neural network modules.

## Features

- **Tensor Operations**: NumPy-compatible array operations
- **Autograd**: Reverse-mode automatic differentiation
- **Neural Network Modules**: Linear, Conv2D, ReLU, etc.
- **Optimizers**: SGD, Adam, RMSprop
- **Pure Python**: No external ML framework dependencies

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from autograd import Tensor, nn, optim

# Define a simple neural network
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Create model and optimizer
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for batch_x, batch_y in dataloader:
    # Forward pass
    pred = model(batch_x)
    loss = nn.cross_entropy(pred, batch_y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Gradient Computation

```python
from autograd import Tensor, grad

# Simple gradient example
x = Tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7.0

# Higher-order gradients
def f(x):
    return x ** 3

df = grad(f)      # First derivative
ddf = grad(df)    # Second derivative

print(df(Tensor([2.0])))   # 12.0
print(ddf(Tensor([2.0])))  # 12.0
```

## Available Modules

| Module | Description |
|--------|-------------|
| `nn.Linear` | Fully connected layer |
| `nn.Conv2D` | 2D convolution |
| `nn.ReLU` | ReLU activation |
| `nn.Sigmoid` | Sigmoid activation |
| `nn.Softmax` | Softmax activation |
| `nn.Dropout` | Dropout regularization |
| `nn.BatchNorm` | Batch normalization |

## Testing

```bash
pytest tests/ -v  # 138 tests
```
