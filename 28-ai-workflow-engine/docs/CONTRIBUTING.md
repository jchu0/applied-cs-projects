# Contributing to AI Workflow Engine

Thank you for your interest in contributing to the AI Workflow Engine! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Contributions](#making-contributions)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discriminatory language, or personal attacks
- Publishing others' private information without permission
- Conduct which could reasonably be considered inappropriate in a professional setting

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, virtualenv, conda)
- Basic understanding of async/await in Python
- Familiarity with workflow orchestration concepts

### Finding Issues to Work On

1. Check the [Issues](https://github.com/your-org/ai-workflow-engine/issues) page
2. Look for issues labeled:
   - `good first issue` - Great for newcomers
   - `help wanted` - We need help with these
   - `enhancement` - New features
   - `bug` - Known bugs to fix

3. Comment on the issue to claim it
4. Wait for maintainer approval before starting work

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/ai-workflow-engine.git
cd ai-workflow-engine

# Add upstream remote
git remote add upstream https://github.com/original-org/ai-workflow-engine.git
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### 4. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aiworkflow --cov-report=html

# Run specific test file
pytest tests/test_engine.py

# Run with verbose output
pytest -vv
```

## Making Contributions

### Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/improvements

### Workflow

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**
   - Go to GitHub and create a PR
   - Fill out the PR template
   - Link related issues

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Test additions or changes
- `chore:` - Build process or auxiliary tool changes

Examples:
```
feat: add support for conditional nodes
fix: resolve memory leak in executor
docs: update API documentation for retry configuration
test: add integration tests for parallel execution
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with these additions:

```python
# Class names: PascalCase
class WorkflowEngine:
    pass

# Function/method names: snake_case
def run_workflow():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 5

# Private methods/attributes: leading underscore
def _internal_method():
    pass
```

### Code Quality

```python
# Use type hints
from typing import Dict, List, Optional

def process_data(
    data: Dict[str, Any],
    filters: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Process data with optional filters.

    Args:
        data: Input data dictionary
        filters: Optional list of filter names

    Returns:
        Processed data dictionary

    Raises:
        ValueError: If data is invalid
    """
    if not data:
        raise ValueError("Data cannot be empty")

    # Implementation
    return processed_data
```

### Async Best Practices

```python
# Use async context managers
async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        data = await response.json()

# Gather concurrent operations
results = await asyncio.gather(
    operation1(),
    operation2(),
    operation3()
)

# Handle exceptions in async code
try:
    result = await async_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise
```

### Error Handling

```python
# Create specific exceptions
class WorkflowValidationError(WorkflowError):
    """Raised when workflow validation fails."""
    pass

# Provide helpful error messages
if not nodes:
    raise WorkflowValidationError(
        "Workflow must contain at least one node. "
        "Please add nodes to your workflow definition."
    )

# Log errors appropriately
try:
    result = execute_node(node)
except RetryableError as e:
    logger.warning(f"Retryable error in node {node.id}: {e}")
    # Retry logic
except Exception as e:
    logger.error(f"Fatal error in node {node.id}: {e}", exc_info=True)
    raise
```

## Testing Guidelines

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

class TestWorkflowEngine:
    """Test suite for WorkflowEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine instance for testing."""
        return WorkflowEngine()

    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.max_parallel == 10
        assert engine.enable_versioning is True

    @pytest.mark.asyncio
    async def test_run_flow(self, engine):
        """Test workflow execution."""
        flow = create_test_flow()
        result = await engine.run_flow(flow)
        assert result.status == RunStatus.COMPLETED

    def test_error_handling(self, engine):
        """Test error handling."""
        with pytest.raises(ValidationError):
            engine.validate_flow(invalid_flow)
```

### Test Coverage Requirements

- Minimum 80% code coverage
- 100% coverage for critical paths
- All public APIs must have tests
- Edge cases and error conditions must be tested

### Running Tests

```bash
# Run all tests with coverage
pytest --cov=aiworkflow --cov-report=term-missing

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Run only integration tests

# Run with parallel execution
pytest -n auto

# Generate HTML coverage report
pytest --cov=aiworkflow --cov-report=html
open htmlcov/index.html
```

## Documentation

### Docstring Format

We use Google-style docstrings:

```python
def execute_workflow(
    flow: FlowDefinition,
    inputs: Dict[str, Any],
    timeout: Optional[float] = None
) -> FlowRun:
    """Execute a workflow with given inputs.

    Args:
        flow: The workflow definition to execute
        inputs: Input data for the workflow
        timeout: Optional timeout in seconds

    Returns:
        FlowRun object containing execution results

    Raises:
        ValidationError: If workflow validation fails
        TimeoutError: If execution exceeds timeout

    Example:
        >>> flow = FlowDefinition(name="test", nodes=[...])
        >>> result = execute_workflow(flow, {"data": [1, 2, 3]})
        >>> print(result.status)
        RunStatus.COMPLETED
    """
```

### Documentation Requirements

- All public functions/methods must have docstrings
- Complex algorithms should have inline comments
- Update README.md for significant changes
- Update API.md for API changes
- Add examples for new features

### Building Documentation

```bash
# Build Sphinx documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## Pull Request Process

### Before Submitting

1. **Update your branch**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**
   ```bash
   pytest
   ```

3. **Run linters**
   ```bash
   # Run all pre-commit hooks
   pre-commit run --all-files

   # Or individually
   black .
   isort .
   flake8 .
   mypy .
   ```

4. **Update documentation**
   - Add/update docstrings
   - Update relevant .md files
   - Add examples if applicable

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings

## Related Issues
Fixes #123
```

### Review Process

1. **Automated Checks**
   - CI/CD pipeline must pass
   - Code coverage maintained
   - No linting errors

2. **Code Review**
   - At least one maintainer approval required
   - Address all feedback
   - Resolve all conversations

3. **Merge**
   - Squash and merge for feature branches
   - Preserve commit history for large features

## Development Tips

### Debugging

```python
# Use logging instead of print
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Processing node: {node.id}")

# Use debugger
import pdb
pdb.set_trace()

# Or with IPython
from IPython import embed
embed()
```

### Performance Profiling

```python
# Profile code execution
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Common Pitfalls

1. **Forgetting await with async functions**
   ```python
   # Wrong
   result = async_function()

   # Correct
   result = await async_function()
   ```

2. **Not handling async context properly**
   ```python
   # Wrong
   with open('file.txt') as f:
       content = f.read()

   # Correct for async
   async with aiofiles.open('file.txt') as f:
       content = await f.read()
   ```

3. **Circular imports**
   - Use TYPE_CHECKING for type hints
   - Restructure modules if necessary

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to AI Workflow Engine!