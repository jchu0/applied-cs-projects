# Contributing to Multi-Tenant GPU Scheduler

Thank you for your interest in contributing to the Multi-Tenant GPU Scheduler project! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Code Style Guidelines](#code-style-guidelines)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Release Process](#release-process)

---

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Expected Behavior

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

### Reporting Issues

If you experience or witness unacceptable behavior, please report it to the project maintainers at gpu-scheduler@example.com.

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. Python 3.8+ installed
2. Git configured with your GitHub account
3. Basic understanding of GPU scheduling concepts
4. Familiarity with Python development

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR-USERNAME/multi-tenant-gpu-scheduler.git
cd multi-tenant-gpu-scheduler
git remote add upstream https://github.com/ORIGINAL-OWNER/multi-tenant-gpu-scheduler.git
```

---

## Development Setup

### 1. Create Development Environment

```bash
# Create virtual environment
python -m venv dev-env
source dev-env/bin/activate  # On Windows: dev-env\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### 2. Configure IDE

**VS Code settings.json:**

```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true,
    "editor.rulers": [88]
}
```

**PyCharm Configuration:**
1. Set Python interpreter to virtual environment
2. Enable pytest as test runner
3. Configure Black as code formatter
4. Enable type checking with mypy

### 3. Development Tools

```bash
# Run tests
pytest tests/

# Run tests with coverage
pytest --cov=gpusched tests/

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
pylint src/

# Type checking
mypy src/

# Run all checks
make check  # Or use tox
```

---

## How to Contribute

### Types of Contributions

#### 1. Bug Reports

Create an issue with:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU type)
- Relevant logs or error messages

**Bug Report Template:**

```markdown
### Description
Brief description of the bug

### Steps to Reproduce
1. Step one
2. Step two
3. ...

### Expected Behavior
What should happen

### Actual Behavior
What actually happens

### Environment
- OS: Ubuntu 20.04
- Python: 3.9.5
- GPU: NVIDIA A100
- CUDA: 11.8

### Logs
```
Error logs here
```
```

#### 2. Feature Requests

Create an issue with:
- Use case description
- Proposed solution
- Alternative solutions considered
- Impact on existing features

#### 3. Code Contributions

- Bug fixes
- New features
- Performance improvements
- Refactoring
- Tests
- Documentation

#### 4. Documentation

- Fix typos or clarify existing docs
- Add examples
- Write tutorials
- Translate documentation

### Finding Issues to Work On

Look for issues labeled:
- `good first issue` - Great for newcomers
- `help wanted` - Community help needed
- `enhancement` - New features
- `bug` - Bug fixes needed
- `documentation` - Documentation improvements

---

## Code Style Guidelines

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these additions:

```python
# Use type hints
def schedule_pod(pod: Pod, cluster: Cluster) -> SchedulingDecision:
    """Schedule a pod to a cluster node.

    Args:
        pod: Pod to schedule
        cluster: Cluster state

    Returns:
        SchedulingDecision with node assignment

    Raises:
        ValueError: If pod requirements invalid
    """
    pass

# Use dataclasses for data structures
@dataclass
class GPUAllocation:
    allocation_id: str
    pod_id: str
    gpu_id: str
    memory_gb: float

# Constants in UPPER_CASE
MAX_GPU_TEMPERATURE = 85.0
DEFAULT_SCHEDULING_INTERVAL = 10

# Private functions/methods with underscore
def _internal_helper():
    pass

# Use descriptive variable names
gpu_utilization_percentage = 75.0  # Good
gpu_util = 75.0  # Avoid
```

### Import Organization

```python
# Standard library imports
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

# Third-party imports
import numpy as np
import pytest

# Local imports
from gpusched.core import resources
from gpusched.scheduler import GPUScheduler
```

### Documentation Standards

```python
def complex_function(
    param1: str,
    param2: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """Brief description (one line).

    Detailed description explaining the function's behavior,
    assumptions, and any important notes.

    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to None.
        **kwargs: Additional keyword arguments:
            - key1: Description
            - key2: Description

    Returns:
        Dictionary containing:
            - 'result': The main result
            - 'metadata': Additional information

    Raises:
        ValueError: If param1 is invalid
        RuntimeError: If operation fails

    Examples:
        >>> result = complex_function("test", param2=42)
        >>> print(result['metadata'])
        {'processed': True}

    Note:
        This function is thread-safe.

    See Also:
        simple_function: A simpler version
    """
    pass
```

---

## Testing Guidelines

### Test Structure

```python
# tests/test_module.py
import pytest
from unittest.mock import Mock, patch

class TestClassName:
    """Test suite for ClassName."""

    def setup_method(self):
        """Setup before each test method."""
        self.fixture = create_test_fixture()

    def teardown_method(self):
        """Cleanup after each test method."""
        cleanup_resources()

    def test_normal_case(self):
        """Test normal operation."""
        # Arrange
        input_data = prepare_input()

        # Act
        result = function_under_test(input_data)

        # Assert
        assert result.success is True
        assert result.value == expected_value

    def test_edge_case(self):
        """Test edge cases."""
        pass

    def test_error_handling(self):
        """Test error scenarios."""
        with pytest.raises(ValueError) as exc_info:
            function_under_test(invalid_input)
        assert "invalid" in str(exc_info.value)

    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_parametrized(self, input, expected):
        """Test with multiple inputs."""
        assert function_under_test(input) == expected
```

### Test Coverage Requirements

- Minimum 80% code coverage for new features
- 100% coverage for critical path code
- Integration tests for major components
- Performance tests for scheduling algorithms

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_scheduler.py

# Run with coverage
pytest --cov=gpusched --cov-report=html

# Run specific test class
pytest tests/test_scheduler.py::TestGPUScheduler

# Run with markers
pytest -m "not slow"

# Parallel execution
pytest -n auto
```

---

## Documentation

### Documentation Requirements

All public APIs must have:
1. Docstrings following Google style
2. Type hints
3. Usage examples
4. Parameter descriptions
5. Return value descriptions
6. Exception descriptions

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build HTML documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

### Adding Documentation

1. API documentation goes in docstrings
2. User guides go in `docs/guides/`
3. Architecture docs go in `docs/architecture/`
4. Examples go in `examples/`

---

## Pull Request Process

### 1. Before Creating PR

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add new scheduling algorithm"

# Run tests
pytest

# Run linters
black src/ tests/
flake8 src/ tests/
mypy src/

# Push to your fork
git push origin feature/your-feature-name
```

### 2. Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): subject

body

footer
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test changes
- `chore`: Build/tool changes

**Examples:**

```bash
feat(scheduler): add gang scheduling support

Implement gang scheduling for distributed training jobs that
require all pods to start simultaneously.

Closes #123

---

fix(allocator): prevent double allocation of GPUs

Fixed race condition in GPU allocation that could cause the
same GPU to be allocated to multiple pods.

Fixes #456
```

### 3. PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Breaking changes documented

## Testing
Describe test cases added/modified

## Screenshots (if applicable)
Add screenshots for UI changes

## Related Issues
Closes #issue_number
```

### 4. Review Process

1. Maintainers will review within 48 hours
2. Address review feedback
3. Ensure CI/CD passes
4. Squash commits if requested
5. PR will be merged after approval

---

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- MAJOR.MINOR.PATCH (e.g., 1.2.3)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist

```bash
# 1. Update version
bump2version minor  # or major/patch

# 2. Update CHANGELOG.md
# Add release notes

# 3. Run full test suite
tox

# 4. Build distribution
python setup.py sdist bdist_wheel

# 5. Test package
pip install dist/gpusched-*.whl

# 6. Tag release
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin v1.2.0

# 7. Create GitHub release
# Upload built packages

# 8. Publish to PyPI
twine upload dist/*
```

---

## Development Tips

### Debugging

```python
# Use logging instead of print
import logging
logger = logging.getLogger(__name__)
logger.debug("Debug information")

# Use debugger
import pdb; pdb.set_trace()

# Profile performance
import cProfile
cProfile.run('scheduler.schedule_pod(pod)')
```

### Performance Optimization

1. Profile before optimizing
2. Focus on algorithmic improvements
3. Use appropriate data structures
4. Consider caching frequently accessed data
5. Write benchmarks for performance-critical code

### Common Pitfalls

1. **Mutable default arguments**
```python
# Wrong
def func(items=[]):
    items.append(1)

# Correct
def func(items=None):
    if items is None:
        items = []
    items.append(1)
```

2. **Not handling GPU unavailability**
```python
# Always check GPU availability
if not gpu.is_available():
    logger.warning(f"GPU {gpu.gpu_id} not available")
    return None
```

3. **Race conditions in scheduling**
```python
# Use proper locking
with scheduler_lock:
    decision = scheduler.schedule_pod(pod)
    allocator.allocate(decision)
```

---

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project website
- Annual contributor report

Thank you for contributing to make GPU scheduling better for everyone!