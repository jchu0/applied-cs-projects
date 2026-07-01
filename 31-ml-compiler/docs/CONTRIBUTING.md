# Contributing to ML Compiler

Thank you for your interest in contributing to ML Compiler! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Process](#contributing-process)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully
- Prioritize the community's best interests

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. Python 3.8 or higher
2. Git for version control
3. A GitHub account
4. Basic understanding of compilers and ML frameworks

### Areas for Contribution

We welcome contributions in the following areas:

- **Core Compiler:** IR design, optimization passes, code generation
- **Backends:** New target hardware support
- **Optimizations:** Performance improvements and new optimization techniques
- **Frontend:** Support for additional ML frameworks
- **Testing:** Test coverage improvements, fuzzing, benchmarks
- **Documentation:** Tutorials, examples, API documentation
- **Tools:** Debugging tools, visualization, profiling

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/ml-compiler.git
cd ml-compiler

# Add upstream remote
git remote add upstream https://github.com/original/ml-compiler.git
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3. Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### 4. Build the Project

```bash
# Build in development mode
python setup.py develop

# Or using pip
pip install -e .

# Build C++ extensions
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### 5. Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_optimization.py

# Run with coverage
pytest --cov=mlcompiler --cov-report=html

# Run benchmarks
pytest benchmarks/ --benchmark-only
```

## Contributing Process

### 1. Find or Create an Issue

Before starting work:

- Check existing issues for something you'd like to work on
- If none exists, create a new issue describing your proposed contribution
- Wait for maintainer feedback before starting major work

### 2. Create a Feature Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `test/` - Test improvements
- `perf/` - Performance improvements

### 3. Make Your Changes

Follow our coding standards and ensure:

- Code is well-documented
- Tests are included for new functionality
- All existing tests pass
- Documentation is updated if needed

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add fusion pass for batch norm and ReLU operations"
git commit -m "Fix memory leak in CUDA code generation"
git commit -m "Improve documentation for IR builder API"

# Bad commit messages (avoid these)
git commit -m "Fix bug"
git commit -m "Update code"
git commit -m "WIP"
```

Commit message format:
```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or fixes
- `perf`: Performance improvements
- `refactor`: Code refactoring
- `style`: Code style changes
- `chore`: Build system or auxiliary tool changes

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:

- Clear title describing the change
- Description of what was changed and why
- Reference to related issues
- Screenshots/examples if applicable

### Pull Request Template

```markdown
## Description
Brief description of changes

## Related Issue
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated documentation

## Screenshots (if applicable)
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some additions:

```python
# File header
"""Module description.

This module provides functionality for...
"""

from typing import List, Optional, Tuple
import numpy as np

# Import order: stdlib, third-party, local

class ClassName:
    """Class description.

    Attributes:
        attribute_name: Description

    Example:
        >>> obj = ClassName()
        >>> obj.method()
    """

    def __init__(self, param: str) -> None:
        """Initialize the class.

        Args:
            param: Parameter description

        Raises:
            ValueError: If param is invalid
        """
        self.param = param

    def method_name(self, arg: int) -> Optional[str]:
        """Method description.

        Args:
            arg: Argument description

        Returns:
            Return value description, None if not found

        Example:
            >>> obj.method_name(42)
            'result'
        """
        # Implementation
        pass
```

### C++ Style Guide

For C++ code, we follow Google C++ Style Guide:

```cpp
// File header
// Copyright notice

#ifndef MLCOMPILER_MODULE_H_
#define MLCOMPILER_MODULE_H_

namespace mlcompiler {

// Class comment
class ClassName {
 public:
  // Constructor
  explicit ClassName(int param);

  // Destructor
  ~ClassName();

  // Public methods
  void MethodName(int arg);

 private:
  // Private members use trailing underscore
  int member_variable_;

  // Private methods
  void PrivateMethod();
};

}  // namespace mlcompiler

#endif  // MLCOMPILER_MODULE_H_
```

### Code Quality Tools

We use the following tools to maintain code quality:

```bash
# Format Python code
black mlcompiler/

# Sort imports
isort mlcompiler/

# Lint Python code
flake8 mlcompiler/
pylint mlcompiler/

# Type checking
mypy mlcompiler/

# Format C++ code
clang-format -i src/**/*.cpp src/**/*.h

# Lint C++ code
cpplint src/**/*.cpp src/**/*.h
```

## Testing Guidelines

### Test Structure

```python
# tests/test_module.py
import unittest
import pytest
from unittest.mock import Mock, patch

class TestClassName(unittest.TestCase):
    """Test cases for ClassName."""

    def setUp(self):
        """Set up test fixtures."""
        self.instance = ClassName()

    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_method_normal_case(self):
        """Test method with normal input."""
        result = self.instance.method(valid_input)
        self.assertEqual(result, expected)

    def test_method_edge_case(self):
        """Test method with edge case."""
        with self.assertRaises(ValueError):
            self.instance.method(invalid_input)

    @pytest.mark.slow
    def test_slow_operation(self):
        """Test that may take longer to run."""
        pass

    @pytest.mark.gpu
    def test_gpu_operation(self):
        """Test requiring GPU."""
        pass
```

### Test Categories

1. **Unit Tests:** Test individual components
2. **Integration Tests:** Test component interactions
3. **End-to-End Tests:** Test complete workflows
4. **Performance Tests:** Benchmark critical paths
5. **Fuzzing Tests:** Random input generation

### Writing Good Tests

```python
# Good test example
def test_convolution_output_shape():
    """Test that convolution produces correct output shape."""
    # Arrange
    input_shape = (32, 3, 224, 224)
    kernel_shape = (64, 3, 7, 7)
    stride = (2, 2)
    padding = (3, 3)

    # Act
    output_shape = compute_conv_output_shape(
        input_shape, kernel_shape, stride, padding
    )

    # Assert
    expected_shape = (32, 64, 112, 112)
    assert output_shape == expected_shape
```

### Test Coverage

Maintain test coverage above 80%:

```bash
# Generate coverage report
pytest --cov=mlcompiler --cov-report=term-missing

# Generate HTML report
pytest --cov=mlcompiler --cov-report=html
```

## Documentation

### Documentation Types

1. **Code Documentation:** Docstrings for all public APIs
2. **User Documentation:** Guides and tutorials
3. **Developer Documentation:** Architecture and design docs
4. **API Reference:** Generated from docstrings

### Writing Documentation

```python
def complex_function(
    param1: str,
    param2: Optional[int] = None,
    **kwargs
) -> Tuple[str, int]:
    """Brief description of function.

    Longer description explaining what the function does,
    when to use it, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to None.
        **kwargs: Additional keyword arguments:
            - key1: Description of key1
            - key2: Description of key2

    Returns:
        A tuple containing:
            - str: Description of first element
            - int: Description of second element

    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not an integer

    Example:
        >>> result = complex_function("input", param2=42)
        >>> print(result)
        ('output', 42)

    Note:
        Additional notes about the function

    See Also:
        related_function: For similar functionality
    """
    pass
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Community

### Communication Channels

- **GitHub Issues:** Bug reports and feature requests
- **GitHub Discussions:** General questions and discussions
- **Discord:** Real-time chat and support
- **Mailing List:** Announcements and discussions

### Recognition

We recognize contributors in several ways:

- Contributors list in README
- Credits in release notes
- Special badges for regular contributors
- Invitation to maintainer team for significant contributors

## Review Process

### Code Review Guidelines

When reviewing code:

1. **Be constructive:** Suggest improvements, don't just criticize
2. **Be specific:** Point to exact lines and suggest alternatives
3. **Be timely:** Try to review within 48 hours
4. **Be thorough:** Check for:
   - Correctness
   - Performance implications
   - Test coverage
   - Documentation
   - Style compliance

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] No unnecessary dependencies added
- [ ] Performance impact considered
- [ ] Security implications reviewed
- [ ] Backward compatibility maintained

## Release Process

### Version Numbering

We use semantic versioning (MAJOR.MINOR.PATCH):

- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible functionality additions
- PATCH: Backwards-compatible bug fixes

### Release Checklist

1. Update version number
2. Update CHANGELOG.md
3. Run full test suite
4. Build and test packages
5. Create release notes
6. Tag release
7. Publish to PyPI
8. Update documentation

## Legal

### License

This project is licensed under the Apache 2.0 License.

### Contributor License Agreement

By contributing, you agree that:

1. You have the right to contribute the code
2. You grant us the right to use your contributions
3. Your contributions are provided "as is"

## Resources

### Useful Links

- [Compiler Design Resources](https://github.com/aalhour/awesome-compilers)
- [ML Systems Papers](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning)
- [LLVM Documentation](https://llvm.org/docs/)
- [MLIR Documentation](https://mlir.llvm.org/)

### Learning Resources

For those new to compiler development:

1. **Books:**
   - "Engineering a Compiler" by Cooper & Torczon
   - "Modern Compiler Implementation" by Appel

2. **Courses:**
   - Stanford CS143: Compilers
   - MIT 6.035: Computer Language Engineering

3. **Papers:**
   - XLA: Domain-specific compiler for linear algebra
   - TVM: An automated end-to-end optimizing compiler

## Thank You!

Thank you for contributing to ML Compiler! Your efforts help make this project better for everyone in the community.