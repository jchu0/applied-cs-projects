# Contributing to Distributed Job Queue

Thank you for your interest in contributing to the Distributed Job Queue project! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Process](#development-process)
4. [Code Style](#code-style)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Reporting Issues](#reporting-issues)

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

## Getting Started

### Prerequisites

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/distributed-job-queue.git
   cd distributed-job-queue
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/original/distributed-job-queue.git
   ```

### Development Environment Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Start Redis for testing**:
   ```bash
   docker run -d -p 6379:6379 --name redis-test redis:7-alpine
   ```

## Development Process

### 1. Branch Strategy

We follow Git Flow branching model:

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Emergency fixes for production
- `release/*` - Release preparation

**Creating a feature branch**:
```bash
git checkout develop
git pull upstream develop
git checkout -b feature/your-feature-name
```

### 2. Making Changes

1. **Keep commits atomic** - Each commit should represent one logical change
2. **Write meaningful commit messages**:
   ```
   feat: Add circuit breaker to worker task processing

   - Implement CircuitBreaker class with configurable thresholds
   - Add automatic retry with exponential backoff
   - Include metrics for circuit breaker state changes

   Closes #123
   ```

3. **Follow conventional commits**:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code style changes (formatting, semicolons, etc.)
   - `refactor:` Code refactoring
   - `perf:` Performance improvements
   - `test:` Test additions or corrections
   - `chore:` Maintenance tasks

### 3. Code Review Process

1. All changes must be submitted via Pull Request
2. PRs must pass all CI checks
3. At least one maintainer approval required
4. Address all review comments
5. Keep PR scope focused and manageable

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good example
from typing import Optional, List, Dict
import asyncio

from jobqueue.models import Task
from jobqueue.broker import Broker


class WorkerPool:
    """Manages a pool of workers for task processing.

    Attributes:
        broker: Connection to the message broker
        workers: List of active worker instances
        max_workers: Maximum number of workers in the pool
    """

    def __init__(
        self,
        broker: Broker,
        max_workers: int = 10,
        queue_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize the worker pool.

        Args:
            broker: Broker instance for task management
            max_workers: Maximum concurrent workers
            queue_names: List of queues to process
        """
        self.broker = broker
        self.max_workers = max_workers
        self.queue_names = queue_names or ["default"]
        self.workers: List[Worker] = []
        self._running = False

    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process a single task.

        Args:
            task: Task to process

        Returns:
            Processing result as dictionary

        Raises:
            TaskProcessingError: If task processing fails
        """
        # Implementation here
        pass
```

### Linting and Formatting

**Required tools:**
- `black` - Code formatter
- `isort` - Import sorter
- `flake8` - Linter
- `mypy` - Type checker
- `pylint` - Advanced linter

**Run all checks:**
```bash
# Format code
black src/ tests/
isort src/ tests/

# Run linters
flake8 src/ tests/
pylint src/
mypy src/

# Or use pre-commit
pre-commit run --all-files
```

### Pre-commit Configuration

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
        additional_dependencies: [types-redis, types-requests]
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Unit tests
│   ├── test_broker.py
│   ├── test_worker.py
│   └── test_models.py
├── integration/       # Integration tests
│   ├── test_end_to_end.py
│   └── test_redis_broker.py
├── performance/       # Performance tests
│   └── test_throughput.py
├── conftest.py       # Shared fixtures
└── fixtures/         # Test data
    └── sample_tasks.json
```

### Writing Tests

**Test example:**
```python
import pytest
from unittest.mock import AsyncMock, patch

from jobqueue.worker import Worker
from jobqueue.models import Task


class TestWorker:
    """Test suite for Worker class."""

    @pytest.fixture
    async def worker(self, mock_broker):
        """Create worker instance for testing."""
        return Worker(broker=mock_broker, concurrency=2)

    @pytest.mark.asyncio
    async def test_process_task_success(self, worker):
        """Test successful task processing."""
        # Arrange
        task = Task(
            id="test-123",
            name="test_task",
            payload={"data": "test"}
        )

        # Act
        result = await worker.process_task(task)

        # Assert
        assert result.status == "completed"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_process_task_with_timeout(self, worker):
        """Test task timeout handling."""
        # Test implementation
        pass
```

### Test Coverage

**Requirements:**
- Minimum 80% code coverage
- 100% coverage for critical paths
- All new features must include tests

**Run tests with coverage:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_worker.py

# Run tests matching pattern
pytest -k "test_process"

# Run only unit tests
pytest tests/unit/

# Run with verbose output
pytest -vv
```

### Performance Testing

```python
# tests/performance/test_throughput.py
import asyncio
import time
from typing import List

import pytest

from jobqueue import Worker, Broker


@pytest.mark.performance
class TestThroughput:
    """Performance test suite."""

    @pytest.mark.asyncio
    async def test_task_throughput(self, broker: Broker):
        """Measure task processing throughput."""
        num_tasks = 1000
        tasks = []

        # Submit tasks
        start_time = time.time()
        for i in range(num_tasks):
            task = await broker.submit_task(
                name="perf_test",
                payload={"index": i}
            )
            tasks.append(task)

        # Wait for completion
        await self.wait_for_tasks(tasks, broker)

        elapsed = time.time() - start_time
        throughput = num_tasks / elapsed

        # Assert performance requirements
        assert throughput > 100, f"Throughput {throughput} below threshold"
```

## Documentation

### Documentation Standards

All code must be well-documented:

1. **Docstrings**: All public modules, classes, and functions
2. **Type hints**: All function parameters and return values
3. **Comments**: Complex logic and algorithms
4. **Examples**: Usage examples in docstrings

**Docstring format (Google style):**
```python
def calculate_retry_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
) -> float:
    """Calculate exponential backoff delay for retries.

    Uses exponential backoff with optional jitter to prevent
    thundering herd problem.

    Args:
        attempt: Current retry attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter: Whether to add random jitter

    Returns:
        Calculated delay in seconds

    Example:
        >>> calculate_retry_delay(0)  # First retry
        1.0
        >>> calculate_retry_delay(3)  # Fourth retry
        8.0

    Note:
        Delay formula: min(base_delay * 2^attempt, max_delay)
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter:
        delay *= random.uniform(0.5, 1.5)
    return delay
```

### API Documentation

Update API documentation when adding/modifying endpoints:

```markdown
## POST /tasks/batch

Submit multiple tasks in a single request.

### Request
```json
{
  "tasks": [
    {
      "name": "task1",
      "payload": {}
    }
  ]
}
```

### Response
```json
{
  "submitted": 2,
  "task_ids": ["id1", "id2"]
}
```

### Errors
- `400`: Invalid task data
- `413`: Too many tasks in batch
```

## Submitting Changes

### Pull Request Process

1. **Update your branch**:
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout feature/your-feature
   git rebase develop
   ```

2. **Run tests locally**:
   ```bash
   pytest
   pre-commit run --all-files
   ```

3. **Push changes**:
   ```bash
   git push origin feature/your-feature
   ```

4. **Create Pull Request**:
   - Use descriptive title
   - Reference related issues
   - Provide detailed description
   - Include test results
   - Add screenshots if applicable

### Pull Request Template

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
Describe test scenarios

## Related Issues
Fixes #123
```

### Review Process

**What reviewers look for:**
1. Code quality and style compliance
2. Test coverage
3. Documentation completeness
4. Performance impact
5. Security considerations
6. Breaking changes

## Reporting Issues

### Bug Reports

**Template:**
```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.0]
- Project version: [e.g., 1.2.0]
- Redis version: [e.g., 7.0.0]

## Additional Context
Logs, screenshots, etc.
```

### Feature Requests

**Template:**
```markdown
## Feature Description
Describe the feature

## Use Case
Why is this needed?

## Proposed Solution
How might this work?

## Alternatives Considered
Other approaches

## Additional Context
Examples, references, etc.
```

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Checklist

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create release branch
6. Tag release
7. Build and publish packages

## Getting Help

### Resources

- **Documentation**: [https://docs.jobqueue.example.com](https://docs.jobqueue.example.com)
- **Discord**: [https://discord.gg/jobqueue](https://discord.gg/jobqueue)
- **Stack Overflow**: Tag with `jobqueue`
- **Email**: dev@jobqueue.example.com

### Community

Join our community:
- Weekly developer meetings (Thursdays 2 PM UTC)
- Monthly contributor spotlight
- Annual contributor summit

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project website
- Annual contributor awards

Thank you for contributing to Distributed Job Queue!