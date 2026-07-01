# Contributing to Dynamic Graph Execution Runtime

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing opinions and experiences

## Getting Started

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/dynamic-graph-runtime.git
   cd dynamic-graph-runtime
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

Follow these guidelines:
- Write clean, readable code
- Add type hints where appropriate
- Follow PEP 8 style guide
- Add docstrings to all functions and classes

### 3. Write Tests

```python
# tests/test_your_feature.py
import unittest
from dynamicgraph import YourFeature

class TestYourFeature(unittest.TestCase):
    def test_functionality(self):
        # Test implementation
        self.assertEqual(expected, actual)
```

Run tests:
```bash
pytest tests/test_your_feature.py
```

### 4. Update Documentation

- Update relevant docs in `docs/`
- Add docstrings to new functions
- Update README if needed

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

Commit message format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or changes
- `refactor:` Code refactoring
- `perf:` Performance improvements

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Create PR on GitHub with:
- Clear description of changes
- Link to related issues
- Test results
- Screenshots if applicable

## Code Style Guidelines

### Python Style

```python
from typing import Optional, List, Dict
import numpy as np

class GraphOptimizer:
    """Optimizer for computation graphs.

    Args:
        optimization_level: Level of optimization (0-3)
        enable_profiling: Whether to enable profiling

    Example:
        >>> optimizer = GraphOptimizer(optimization_level=2)
        >>> optimized = optimizer.optimize(graph)
    """

    def __init__(
        self,
        optimization_level: int = 1,
        enable_profiling: bool = False
    ) -> None:
        self.optimization_level = optimization_level
        self.enable_profiling = enable_profiling

    def optimize(self, graph: Graph) -> Graph:
        """Optimize the given graph.

        Args:
            graph: Input computation graph

        Returns:
            Optimized graph

        Raises:
            ValueError: If graph is invalid
        """
        if not graph.validate():
            raise ValueError("Invalid graph")

        # Implementation
        return optimized_graph
```

### Testing Guidelines

```python
class TestGraphOptimizer(unittest.TestCase):
    """Tests for GraphOptimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = GraphOptimizer()
        self.test_graph = create_test_graph()

    def test_optimization_preserves_semantics(self):
        """Test that optimization preserves graph semantics."""
        original_result = execute(self.test_graph)
        optimized = self.optimizer.optimize(self.test_graph)
        optimized_result = execute(optimized)

        np.testing.assert_allclose(original_result, optimized_result)

    def test_invalid_graph_raises_error(self):
        """Test that invalid graphs raise appropriate errors."""
        invalid_graph = create_invalid_graph()

        with self.assertRaises(ValueError):
            self.optimizer.optimize(invalid_graph)
```

## Project Structure

```
dynamic-graph-runtime/
├── src/dynamicgraph/        # Source code
│   ├── core/                # Core functionality
│   ├── tracer/              # Tracing components
│   ├── ir/                  # IR representation
│   ├── optimizer/           # Optimization passes
│   └── codegen/             # Code generation
├── tests/                   # Test files
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── benchmarks/         # Performance tests
├── docs/                    # Documentation
├── examples/                # Example code
└── scripts/                 # Utility scripts
```

## Adding New Features

### 1. New Operation Type

```python
# src/dynamicgraph/core/ops.py
class NewOperation(Operation):
    def forward(self, inputs):
        # Forward implementation
        pass

    def backward(self, grad_output):
        # Backward implementation
        pass

# Register the operation
register_op("new_op", NewOperation)
```

### 2. New Optimization Pass

```python
# src/dynamicgraph/optimizer/passes/new_pass.py
class NewOptimizationPass(OptimizationPass):
    def apply(self, graph: Graph) -> Graph:
        # Transform graph
        for node in graph.nodes():
            if self.can_optimize(node):
                self.optimize_node(node)
        return graph
```

### 3. New Backend

```python
# src/dynamicgraph/codegen/backends/new_backend.py
class NewBackend(Backend):
    def compile(self, graph: Graph) -> CompiledCode:
        # Compilation logic
        pass

    def execute(self, code: CompiledCode, inputs: Dict) -> Any:
        # Execution logic
        pass
```

## Testing Requirements

### Coverage Requirements

- Minimum 80% code coverage for new features
- 100% coverage for critical paths
- All edge cases tested

### Test Categories

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Benchmark performance
4. **Regression Tests**: Prevent regressions

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dynamicgraph --cov-report=html

# Run specific test file
pytest tests/test_specific.py

# Run with verbose output
pytest -v

# Run performance benchmarks
pytest tests/benchmarks/ --benchmark-only
```

## Documentation

### Docstring Format

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """Brief description of function.

    Longer description if needed, explaining the purpose
    and behavior of the function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ExceptionType: When this exception is raised

    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
        expected_output

    Note:
        Additional notes about the function
    """
    pass
```

## Release Process

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Run full test suite
4. Create release branch
5. Create GitHub release
6. Deploy to PyPI

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation