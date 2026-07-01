# Contributing to HDFS-Like Distributed File System

Thank you for your interest in contributing to our HDFS implementation! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We expect all participants to:

- Be respectful and considerate
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information without permission
- Any conduct which would be considered inappropriate in a professional setting

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of distributed systems
- Familiarity with HDFS concepts

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR-USERNAME/hdfs-python.git
cd hdfs-python
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/ORIGINAL-OWNER/hdfs-python.git
```

## Development Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install the package in development mode
pip install -e .
```

### 3. Set Up Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install
```

### 4. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hdfs --cov-report=html

# Run specific test file
pytest tests/test_namenode.py
```

## How to Contribute

### Types of Contributions

#### 1. Bug Reports

- Use the issue tracker to report bugs
- Provide detailed information about the bug
- Include steps to reproduce
- Mention your environment (OS, Python version, etc.)

#### 2. Bug Fixes

- Look for issues labeled `bug` or `good first issue`
- Write tests that demonstrate the bug is fixed
- Update documentation if needed

#### 3. Feature Development

- Discuss new features in an issue first
- Implement features following our coding standards
- Write comprehensive tests
- Update documentation

#### 4. Documentation

- Fix typos and improve clarity
- Add examples and tutorials
- Update API documentation
- Translate documentation

#### 5. Performance Improvements

- Profile code to identify bottlenecks
- Provide benchmarks showing improvements
- Ensure no regression in functionality

## Development Workflow

### 1. Create a Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

```bash
# Make your changes
vim src/hdfs/your_file.py

# Run tests frequently
pytest tests/

# Check code style
flake8 src/hdfs/
black --check src/hdfs/
```

### 3. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add support for erasure coding

- Implement Reed-Solomon erasure coding
- Add configuration for EC policies
- Update tests for EC functionality

Closes #123"
```

#### Commit Message Format

We follow the Conventional Commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or changes
- `chore`: Maintenance tasks

### 4. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Coding Standards

### Python Style Guide

We follow PEP 8 with some additions:

```python
# Good: Descriptive variable names
block_replication_factor = 3

# Bad: Single letter variables (except in loops)
b = 3

# Good: Type hints
def allocate_blocks(
    path: str,
    num_blocks: int,
    block_size: int = 128 * 1024 * 1024
) -> List[Block]:
    pass

# Good: Comprehensive docstrings
def get_block_locations(self, block_id: BlockID) -> List[BlockLocation]:
    """
    Get the locations where a block is stored.

    Args:
        block_id: The unique identifier of the block

    Returns:
        List of BlockLocation objects containing DataNode information

    Raises:
        BlockNotFoundError: If the block doesn't exist
    """
    pass
```

### Code Organization

```
src/hdfs/
├── common/          # Shared utilities and types
│   ├── __init__.py
│   ├── types.py     # Type definitions
│   └── protocol.py  # Protocol definitions
├── namenode/        # NameNode implementation
│   ├── __init__.py
│   └── namenode.py
├── datanode/        # DataNode implementation
│   ├── __init__.py
│   └── datanode.py
└── client/          # Client implementation
    ├── __init__.py
    └── client.py
```

### Async/Await Guidelines

```python
# Good: Use async/await for I/O operations
async def read_block(self, block_id: BlockID) -> bytes:
    async with aiofiles.open(self.get_block_path(block_id), 'rb') as f:
        return await f.read()

# Good: Use asyncio.gather for parallel operations
async def read_multiple_blocks(self, block_ids: List[BlockID]) -> List[bytes]:
    tasks = [self.read_block(bid) for bid in block_ids]
    return await asyncio.gather(*tasks)
```

### Error Handling

```python
# Good: Specific exception types
class BlockNotFoundError(HDFSError):
    """Raised when a requested block doesn't exist."""
    pass

# Good: Informative error messages
if not os.path.exists(block_path):
    raise BlockNotFoundError(
        f"Block {block_id} not found at {block_path}"
    )

# Good: Proper cleanup in finally blocks
try:
    reader, writer = await asyncio.open_connection(host, port)
    # ... do work ...
finally:
    writer.close()
    await writer.wait_closed()
```

## Testing Guidelines

### Test Structure

```python
import pytest
from unittest.mock import Mock, AsyncMock

class TestNameNode:
    """Test cases for NameNode functionality."""

    @pytest.fixture
    def namenode(self):
        """Create a NameNode instance for testing."""
        return NameNode(default_replication=3)

    def test_create_file(self, namenode):
        """Test file creation."""
        # Arrange
        path = "/test.txt"

        # Act
        file_info = namenode.create_file(path)

        # Assert
        assert file_info.path == path
        assert path in namenode._files

    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async operations."""
        # ... async test code ...
```

### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Include integration tests
- Add performance tests for critical paths

### Test Data

```python
# Good: Use fixtures for test data
@pytest.fixture
def sample_blocks():
    """Generate sample blocks for testing."""
    return [
        Block(block_id=f"blk_{i}", size=1024)
        for i in range(10)
    ]

# Good: Use parameterized tests
@pytest.mark.parametrize("replication,expected", [
    (1, 1),
    (3, 3),
    (5, 5),
])
def test_replication_factor(replication, expected):
    # ... test code ...
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def complex_function(
    param1: str,
    param2: int,
    param3: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Brief description of the function.

    Longer description if needed, explaining the purpose
    and behavior of the function in more detail.

    Args:
        param1: Description of param1
        param2: Description of param2
        param3: Description of param3. Defaults to None.

    Returns:
        Description of the return value, including structure
        for complex types.

    Raises:
        ValueError: When param2 is negative
        TypeError: When param1 is not a string

    Example:
        >>> result = complex_function("test", 42)
        >>> print(result['status'])
        'success'

    Note:
        Any additional notes about the function.
    """
    pass
```

### API Documentation

- Update `docs/API.md` for public API changes
- Include code examples
- Document all parameters and return values
- Explain error conditions

### Architecture Documentation

- Update `docs/ARCHITECTURE.md` for design changes
- Include diagrams where helpful
- Explain design decisions and trade-offs

## Pull Request Process

### Before Submitting

1. **Update your branch**:
```bash
git fetch upstream
git rebase upstream/main
```

2. **Run all checks**:
```bash
# Run tests
pytest

# Check code style
flake8 src/hdfs/
black src/hdfs/
mypy src/hdfs/

# Check documentation
pydoc src/hdfs
```

3. **Update documentation**:
- Add docstrings for new functions
- Update README if needed
- Add to CHANGELOG.md

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Changes Made
- List specific changes
- Include files modified

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Added new tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings

## Related Issues
Fixes #123
```

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address all review comments
4. Squash commits if requested
5. Maintainer merges when ready

## Issue Guidelines

### Bug Reports

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
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.8.10]
- Version: [e.g., 1.0.0]

## Additional Context
Any other relevant information
```

### Feature Requests

```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed?

## Proposed Solution
How would you implement it?

## Alternatives Considered
Other approaches you've thought about

## Additional Context
Any other relevant information
```

## Release Process

### Version Numbering

We use Semantic Versioning (SemVer):
- MAJOR.MINOR.PATCH
- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

### Release Checklist

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Run full test suite
4. Build documentation
5. Tag release
6. Create GitHub release
7. Deploy to PyPI

## Community

### Communication Channels

- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General discussions
- Slack: Real-time chat (invite link in README)
- Mailing List: Major announcements

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (Apache 2.0).