# Contributing to Vector Quantized LLM

Thank you for your interest in contributing to the Vector Quantized LLM project! This document provides guidelines and information for contributors.

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
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Expected Behavior

- Demonstrate empathy and kindness toward other people
- Be respectful of differing opinions, viewpoints, and experiences
- Give and gracefully accept constructive feedback
- Accept responsibility for mistakes and learn from the experience
- Focus on what is best for the overall community

### Unacceptable Behavior

- Harassment, discriminatory language, or personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Conduct which would be considered inappropriate in a professional setting

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. Python 3.8 or later installed
2. Git configured with your GitHub account
3. Familiarity with quantization concepts (recommended)
4. Basic understanding of transformer models

### Setting Up Your Fork

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/YOUR_USERNAME/vector-quantized-llm.git
cd vector-quantized-llm
```

3. Add the upstream repository:
```bash
git remote add upstream https://github.com/original-org/vector-quantized-llm.git
```

4. Create a branch for your changes:
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

# Key development tools
# - pytest: Testing framework
# - black: Code formatter
# - flake8: Linter
# - mypy: Type checker
# - sphinx: Documentation generator
# - pre-commit: Git hooks
```

### IDE Configuration

#### VS Code

`.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "editor.formatOnSave": true,
    "editor.rulers": [88]
}
```

#### PyCharm

1. Set Python interpreter to virtual environment
2. Enable Black formatter: Settings → Tools → External Tools
3. Configure pytest: Settings → Tools → Python Integrated Tools
4. Set line length to 88 characters

## How to Contribute

### Types of Contributions

#### 1. Bug Reports

Report bugs via GitHub Issues. Include:

- Description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- System information (OS, Python version, GPU)
- Relevant logs or error messages

**Template:**
```markdown
### Bug Description
[Clear description of the bug]

### Reproduction Steps
1. [First step]
2. [Second step]
3. [...]

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Environment
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- CUDA: [e.g., 11.8]
- Package version: [e.g., 0.1.0]

### Logs
```
[Relevant error messages or logs]
```
```

#### 2. Feature Requests

Suggest features via GitHub Issues. Include:

- Use case description
- Proposed solution
- Alternative solutions considered
- Impact on existing functionality

**Template:**
```markdown
### Feature Description
[Clear description of the feature]

### Use Case
[Why this feature is needed]

### Proposed Solution
[How you envision this working]

### Alternatives Considered
[Other approaches you've thought about]

### Additional Context
[Any other relevant information]
```

#### 3. Code Contributions

Areas where contributions are welcome:

- **New Quantization Methods**: Implement novel quantization techniques
- **Performance Optimizations**: Improve inference speed or memory usage
- **Hardware Support**: Add support for new hardware accelerators
- **Bug Fixes**: Fix reported issues
- **Documentation**: Improve or translate documentation
- **Tests**: Increase test coverage

#### 4. Documentation

Help improve documentation:

- Fix typos or clarify explanations
- Add examples and tutorials
- Translate documentation
- Create video tutorials or blog posts

## Code Style Guidelines

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Maximum line length: 88 characters (Black default)
# Use double quotes for strings
# Use type hints for public functions

from typing import Optional, List, Dict, Any
import numpy as np

class QuantizerExample:
    """Example quantizer implementation.

    Args:
        config: Quantization configuration
        device: Device to run on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu"
    ) -> None:
        self.config = config or {}
        self.device = device

    def quantize(
        self,
        tensor: np.ndarray,
        name: str = ""
    ) -> "QuantizedTensor":
        """Quantize a tensor.

        Args:
            tensor: Input tensor to quantize
            name: Optional name for logging

        Returns:
            QuantizedTensor object

        Raises:
            ValueError: If tensor shape is invalid
        """
        if tensor.ndim < 2:
            raise ValueError(f"Tensor must be at least 2D, got {tensor.ndim}D")

        # Implementation here
        return self._do_quantization(tensor, name)

    def _do_quantization(
        self,
        tensor: np.ndarray,
        name: str
    ) -> "QuantizedTensor":
        """Private method for actual quantization."""
        # Note: Use single underscore for private methods
        pass
```

### Naming Conventions

```python
# Classes: PascalCase
class MyQuantizer:
    pass

# Functions and variables: snake_case
def compute_scale_factor():
    scale_factor = 1.0
    return scale_factor

# Constants: UPPER_SNAKE_CASE
MAX_BATCH_SIZE = 128
DEFAULT_BLOCK_SIZE = 64

# Private attributes: single underscore prefix
class Example:
    def __init__(self):
        self._private_attr = None

# Module-level private: single underscore prefix
_private_helper_function = lambda x: x * 2
```

### Import Organization

```python
# Standard library imports
import os
import sys
from typing import Optional, List

# Third-party imports
import numpy as np
import torch

# Local imports
from vqllm.core import QuantConfig
from vqllm.quantize import Quantizer
```

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str = "default") -> bool:
    """Brief description of function.

    Longer description if needed. Can span multiple lines
    and include examples.

    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to "default".

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this error is raised

    Examples:
        >>> result = example_function(42, "test")
        >>> print(result)
        True

    Note:
        Additional notes or warnings
    """
    pass
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                 # Unit tests
│   ├── test_quantizers.py
│   ├── test_calibration.py
│   └── test_inference.py
├── integration/          # Integration tests
│   ├── test_pipeline.py
│   └── test_models.py
├── performance/          # Performance tests
│   ├── test_benchmarks.py
│   └── test_memory.py
├── fixtures/            # Test fixtures and data
│   ├── __init__.py
│   └── sample_data.py
└── conftest.py          # Pytest configuration
```

### Writing Tests

```python
# test_quantizer.py
import pytest
import numpy as np
from vqllm.quantize import INT8Quantizer

class TestINT8Quantizer:
    """Test INT8 quantization functionality."""

    @pytest.fixture
    def quantizer(self):
        """Create quantizer instance."""
        return INT8Quantizer()

    @pytest.fixture
    def sample_weight(self):
        """Generate sample weight tensor."""
        return np.random.randn(256, 256).astype(np.float32)

    def test_quantize_weight(self, quantizer, sample_weight):
        """Test basic weight quantization."""
        # Arrange
        expected_dtype = np.int8

        # Act
        qtensor = quantizer.quantize_weight(sample_weight)

        # Assert
        assert qtensor.dtype == expected_dtype
        assert qtensor.shape == sample_weight.shape
        assert qtensor.scale is not None

    @pytest.mark.parametrize("shape", [
        (128, 128),
        (256, 512),
        (1024, 1024)
    ])
    def test_different_shapes(self, quantizer, shape):
        """Test quantization with different tensor shapes."""
        weight = np.random.randn(*shape).astype(np.float32)
        qtensor = quantizer.quantize_weight(weight)
        assert qtensor.shape == shape

    def test_quantization_error(self, quantizer, sample_weight):
        """Test quantization error is within bounds."""
        qtensor = quantizer.quantize_weight(sample_weight)
        dequant = qtensor.dequantize()

        # Calculate error
        error = np.mean(np.abs(sample_weight - dequant))
        relative_error = error / np.mean(np.abs(sample_weight))

        # Should have < 5% relative error for INT8
        assert relative_error < 0.05

    @pytest.mark.slow
    def test_large_model(self, quantizer):
        """Test quantization of large model (marked as slow)."""
        # This test is skipped unless --runslow is passed
        large_weight = np.random.randn(4096, 4096).astype(np.float32)
        qtensor = quantizer.quantize_weight(large_weight)
        assert qtensor is not None
```

### Test Coverage

Aim for minimum 80% test coverage:

```bash
# Run tests with coverage
pytest --cov=vqllm --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html  # On macOS
# xdg-open htmlcov/index.html  # On Linux

# Check coverage meets minimum
pytest --cov=vqllm --cov-fail-under=80
```

### Performance Testing

```python
# test_performance.py
import pytest
import time
import numpy as np
from vqllm.quantize import INT8Quantizer

@pytest.mark.benchmark
class TestPerformance:
    """Performance regression tests."""

    def test_quantization_speed(self, benchmark):
        """Test quantization performance."""
        quantizer = INT8Quantizer()
        weight = np.random.randn(1024, 1024).astype(np.float32)

        # Benchmark the quantization
        result = benchmark(quantizer.quantize_weight, weight)

        # Assert performance requirements
        assert benchmark.stats["mean"] < 0.1  # < 100ms

    def test_memory_usage(self):
        """Test memory efficiency."""
        import tracemalloc

        quantizer = INT8Quantizer()
        weight = np.random.randn(1024, 1024).astype(np.float32)

        tracemalloc.start()
        qtensor = quantizer.quantize_weight(weight)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be less than 2x original
        original_size = weight.nbytes
        assert peak < original_size * 2
```

## Documentation Guidelines

### Documentation Structure

```
docs/
├── source/
│   ├── conf.py          # Sphinx configuration
│   ├── index.rst        # Main documentation index
│   ├── quickstart.rst   # Getting started guide
│   ├── api/             # API reference
│   │   ├── quantizers.rst
│   │   └── inference.rst
│   ├── tutorials/       # Tutorials
│   │   ├── basic_usage.rst
│   │   └── advanced.rst
│   └── developer/       # Developer documentation
│       ├── architecture.rst
│       └── contributing.rst
├── build/              # Generated documentation
└── Makefile           # Build commands
```

### Writing Documentation

```rst
.. _quantizer-api:

Quantizer API
=============

This section describes the quantizer API.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The quantizer module provides various quantization methods...

.. note::
   This is an important note for users.

.. warning::
   This is a warning about potential issues.

API Reference
-------------

.. autoclass:: vqllm.quantize.INT8Quantizer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Examples
--------

Basic usage example::

    from vqllm.quantize import INT8Quantizer

    quantizer = INT8Quantizer()
    qtensor = quantizer.quantize_weight(weight)

Advanced example:

.. code-block:: python
   :linenos:

   config = QuantConfig(
       quant_type="int8",
       scale_type="per_channel"
   )
   quantizer = INT8Quantizer(config)

.. seealso::
   :ref:`calibration-api` for calibration details
```

### Building Documentation

```bash
# Build HTML documentation
cd docs
make html

# Build PDF documentation (requires LaTeX)
make latexpdf

# Check for broken links
make linkcheck

# Clean build artifacts
make clean
```

## Pull Request Process

### Before Submitting

1. **Update your branch**:
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
black --check vqllm/
flake8 vqllm/
mypy vqllm/
```

4. **Update documentation** if needed

5. **Add tests** for new features

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] PR description explains changes
- [ ] No merge conflicts
- [ ] Tests added for new features
- [ ] Performance impact considered

### PR Template

```markdown
## Description
[Describe your changes]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Other (specify)

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guide
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes

## Related Issues
Fixes #[issue number]

## Screenshots (if applicable)
[Add screenshots]
```

### Review Process

1. **Automatic checks**: CI/CD runs tests and linters
2. **Code review**: At least one maintainer reviews
3. **Discussion**: Address feedback and questions
4. **Approval**: Maintainer approves
5. **Merge**: Maintainer merges PR

## Release Process

### Version Numbering

We follow Semantic Versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Release Checklist

1. **Update version**:
```python
# vqllm/__init__.py
__version__ = "0.2.0"
```

2. **Update CHANGELOG**:
```markdown
## [0.2.0] - 2024-01-15
### Added
- New AWQ quantization method
- Multi-GPU support

### Fixed
- Memory leak in KV cache

### Changed
- Improved INT4 performance
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

6. **Test installation**:
```bash
pip install dist/vqllm-0.2.0-py3-none-any.whl
```

7. **Tag release**:
```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

8. **Upload to PyPI**:
```bash
twine upload dist/*
```
