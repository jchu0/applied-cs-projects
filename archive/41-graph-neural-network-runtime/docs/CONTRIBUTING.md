# Contributing to Graph Neural Network Runtime

Thank you for your interest in contributing to the GNN Runtime project! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Code Style Guidelines](#code-style-guidelines)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation Guidelines](#documentation-guidelines)
8. [Pull Request Process](#pull-request-process)
9. [Release Process](#release-process)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We pledge to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards others

### Expected Behavior

- Demonstrate kindness and empathy
- Respect differing opinions and experiences
- Give and accept constructive feedback gracefully
- Accept responsibility and learn from mistakes
- Focus on community benefit

### Unacceptable Behavior

- Harassment or discriminatory language
- Personal attacks or insults
- Publishing private information without permission
- Unprofessional conduct

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. Python 3.8+ installed
2. Git configured
3. Understanding of graph theory basics
4. Familiarity with GNN concepts

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/gnn-runtime.git
cd gnn-runtime
```

3. Add upstream remote:
```bash
git remote add upstream https://github.com/original-org/gnn-runtime.git
```

4. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

## Development Setup

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify setup
python -m pytest tests/test_basic.py
```

### Development Dependencies

```bash
# Install all development dependencies
pip install -r requirements-dev.txt

# Key development tools:
# - pytest: Testing framework
# - black: Code formatter
# - flake8: Linter
# - mypy: Type checker
# - sphinx: Documentation
# - coverage: Code coverage
```

### IDE Configuration

#### VS Code

`.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "editor.formatOnSave": true,
    "editor.rulers": [88],
    "python.linting.mypyEnabled": true
}
```

#### PyCharm

1. Set interpreter to virtual environment
2. Enable Black formatter
3. Configure pytest as test runner
4. Set line length to 88

## How to Contribute

### Types of Contributions

#### 1. Bug Reports

Report bugs via GitHub Issues with:

- Clear bug description
- Steps to reproduce
- Expected vs actual behavior
- System information
- Error messages/logs

**Template:**
```markdown
### Bug Description
[Clear description]

### Reproduction Steps
1. [Step 1]
2. [Step 2]
...

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Environment
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- GNN Runtime version: [e.g., 0.1.0]
- GPU: [e.g., NVIDIA RTX 3090]

### Logs
```
[Error messages]
```
```

#### 2. Feature Requests

Suggest features via GitHub Issues with:

- Use case description
- Proposed solution
- Alternative approaches
- Impact assessment

**Template:**
```markdown
### Feature Description
[Clear description]

### Use Case
[Why this feature is needed]

### Proposed Implementation
[How it could work]

### Alternatives
[Other approaches considered]

### Impact
[Effect on existing functionality]
```

#### 3. Code Contributions

Areas where contributions are welcome:

- **New GNN Layers**: Implement novel architectures
- **Sampling Methods**: Add new sampling strategies
- **Performance**: Optimize existing code
- **Bug Fixes**: Fix reported issues
- **Documentation**: Improve docs and examples
- **Tests**: Increase test coverage

#### 4. Documentation

Help improve documentation:

- Fix typos and clarify text
- Add examples and tutorials
- Create visual diagrams
- Translate documentation

## Code Style Guidelines

### Python Style Guide

Follow PEP 8 with Black formatting:

```python
# Maximum line length: 88 characters
# Use type hints for public functions
# Use docstrings for all public APIs

from typing import Optional, List, Tuple, Union
import numpy as np


class GNNLayer:
    """Base class for GNN layers.

    Args:
        in_channels: Input feature dimensionality
        out_channels: Output feature dimensionality
        aggr: Aggregation method ('add', 'mean', 'max')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "add"
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

    def forward(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        edge_attr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass through the layer.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Node embeddings [num_nodes, out_channels]

        Raises:
            ValueError: If input shapes are invalid
        """
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"got {x.shape[1]}"
            )

        return self._propagate(x, edge_index, edge_attr)

    def _propagate(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        edge_attr: Optional[np.ndarray]
    ) -> np.ndarray:
        """Internal propagation method."""
        # Implementation here
        pass
```

### Naming Conventions

```python
# Classes: PascalCase
class GraphConvolution:
    pass

# Functions and variables: snake_case
def compute_edge_weights():
    edge_weights = []
    return edge_weights

# Constants: UPPER_SNAKE_CASE
DEFAULT_NUM_NEIGHBORS = 25
MAX_BATCH_SIZE = 1024

# Private methods: single underscore prefix
def _internal_helper():
    pass

# Module-level private: single underscore prefix
_cache = {}
```

### Import Organization

```python
# Standard library
import os
import sys
from typing import Optional, List

# Third-party
import numpy as np
import scipy.sparse as sp

# Local imports
from gnnruntime.core import Graph
from gnnruntime.layers import GCNConv
```

### Docstring Format

Use Google-style docstrings:

```python
def sample_neighbors(
    node_idx: int,
    edge_index: np.ndarray,
    num_samples: int = 25,
    replace: bool = False
) -> np.ndarray:
    """Sample neighbors for a given node.

    Samples a fixed number of neighbors uniformly at random
    from the node's neighborhood.

    Args:
        node_idx: Index of the target node
        edge_index: Graph connectivity [2, num_edges]
        num_samples: Number of neighbors to sample
        replace: Whether to sample with replacement

    Returns:
        Array of sampled neighbor indices

    Raises:
        ValueError: If node_idx is out of bounds

    Examples:
        >>> edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        >>> neighbors = sample_neighbors(0, edge_index, 2)
        >>> print(neighbors)
        [1, 2]

    Note:
        For nodes with fewer neighbors than num_samples,
        all neighbors are returned.
    """
    pass
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                 # Unit tests
│   ├── test_layers.py
│   ├── test_sampling.py
│   └── test_data.py
├── integration/         # Integration tests
│   ├── test_pipeline.py
│   └── test_models.py
├── performance/         # Performance tests
│   ├── test_benchmarks.py
│   └── test_scalability.py
├── fixtures/           # Test fixtures
│   ├── __init__.py
│   └── graph_data.py
└── conftest.py         # Pytest configuration
```

### Writing Tests

```python
# test_gcn_layer.py
import pytest
import numpy as np
from gnnruntime.layers import GCNConv


class TestGCNConv:
    """Test GCN convolution layer."""

    @pytest.fixture
    def layer(self):
        """Create GCN layer."""
        return GCNConv(16, 32)

    @pytest.fixture
    def graph_data(self):
        """Create sample graph data."""
        x = np.random.randn(100, 16).astype(np.float32)
        edge_index = np.random.randint(0, 100, (2, 500))
        return x, edge_index

    def test_forward_pass(self, layer, graph_data):
        """Test forward pass."""
        x, edge_index = graph_data
        output = layer(x, edge_index)

        assert output.shape == (100, 32)
        assert output.dtype == np.float32

    @pytest.mark.parametrize("in_dim,out_dim", [
        (8, 16),
        (16, 32),
        (32, 64)
    ])
    def test_dimensions(self, in_dim, out_dim):
        """Test different dimensions."""
        layer = GCNConv(in_dim, out_dim)
        x = np.random.randn(50, in_dim).astype(np.float32)
        edge_index = np.random.randint(0, 50, (2, 100))

        output = layer(x, edge_index)
        assert output.shape == (50, out_dim)

    def test_invalid_input(self, layer):
        """Test invalid input handling."""
        x = np.random.randn(10, 8)  # Wrong input dimension
        edge_index = np.array([[0, 1], [1, 0]])

        with pytest.raises(ValueError):
            layer(x, edge_index)

    @pytest.mark.slow
    def test_large_graph(self):
        """Test on large graph."""
        layer = GCNConv(128, 256)
        x = np.random.randn(10000, 128).astype(np.float32)
        edge_index = np.random.randint(0, 10000, (2, 50000))

        output = layer(x, edge_index)
        assert output.shape == (10000, 256)
```

### Test Coverage

Aim for 80%+ test coverage:

```bash
# Run tests with coverage
pytest --cov=gnnruntime --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html

# Check coverage threshold
pytest --cov=gnnruntime --cov-fail-under=80
```

### Performance Testing

```python
# test_performance.py
import pytest
import time
import numpy as np


@pytest.mark.benchmark
def test_gcn_performance(benchmark):
    """Benchmark GCN layer."""
    from gnnruntime.layers import GCNConv

    layer = GCNConv(128, 256)
    x = np.random.randn(1000, 128).astype(np.float32)
    edge_index = np.random.randint(0, 1000, (2, 5000))

    result = benchmark(layer, x, edge_index)
    assert result.shape == (1000, 256)

    # Check performance
    assert benchmark.stats["mean"] < 0.1  # < 100ms
```

## Documentation Guidelines

### Documentation Structure

```
docs/
├── source/
│   ├── conf.py          # Sphinx configuration
│   ├── index.rst        # Main index
│   ├── api/             # API reference
│   │   ├── layers.rst
│   │   └── sampling.rst
│   ├── tutorials/       # Tutorials
│   │   ├── quickstart.rst
│   │   └── advanced.rst
│   └── developer/       # Developer docs
│       └── architecture.rst
├── build/              # Generated docs
└── Makefile
```

### Writing Documentation

```rst
.. _gcn-layer:

GCN Layer
=========

Graph Convolutional Network layer implementation.

.. contents::
   :local:
   :depth: 2

Overview
--------

The GCN layer implements spectral-based graph convolution...

.. math::
   H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})

API Reference
-------------

.. autoclass:: gnnruntime.layers.GCNConv
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Basic usage::

    from gnnruntime.layers import GCNConv

    layer = GCNConv(16, 32)
    output = layer(x, edge_index)

With edge weights:

.. code-block:: python
   :linenos:

   layer = GCNConv(16, 32)
   output = layer(x, edge_index, edge_weight=weights)

.. note::
   Edge weights must be non-negative.

.. seealso::
   :ref:`gat-layer` for attention-based convolution
```

### Building Documentation

```bash
# Build HTML docs
cd docs
make html

# Build PDF docs
make latexpdf

# Check for broken links
make linkcheck

# Clean build
make clean
```

## Pull Request Process

### Before Submitting

1. **Update branch**:
```bash
git fetch upstream
git rebase upstream/main
```

2. **Run tests**:
```bash
pytest tests/
```

3. **Check code style**:
```bash
black --check gnnruntime/
flake8 gnnruntime/
mypy gnnruntime/
```

4. **Update documentation**

5. **Add tests for new features**

### PR Template

```markdown
## Description
[Describe changes]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass

## Checklist
- [ ] Code follows style guide
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes

## Related Issues
Fixes #[issue number]
```

### Review Process

1. Automated CI checks
2. Code review by maintainer
3. Address feedback
4. Approval and merge

## Release Process

### Versioning

Follow Semantic Versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes

### Release Checklist

1. **Update version**:
```python
# gnnruntime/__init__.py
__version__ = "0.2.0"
```

2. **Update CHANGELOG**:
```markdown
## [0.2.0] - 2024-01-15
### Added
- New sampling methods
- GPU acceleration

### Fixed
- Memory leak in batching

### Changed
- Improved GCN performance
```

3. **Create release branch**:
```bash
git checkout -b release/0.2.0
```

4. **Run full test suite**:
```bash
pytest --runslow tests/
```

5. **Build distribution**:
```bash
python setup.py sdist bdist_wheel
```

6. **Tag release**:
```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

7. **Upload to PyPI**:
```bash
twine upload dist/*
```

## Questions and Support

### Getting Help

- **GitHub Discussions**: Ask questions
- **Discord**: Join our community
- **Stack Overflow**: Tag with `gnn-runtime`

### Contact

- General: community@gnnruntime.ai
- Security: security@gnnruntime.ai

Thank you for contributing to GNN Runtime!