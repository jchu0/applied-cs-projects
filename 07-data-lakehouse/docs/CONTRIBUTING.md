# Contributing to Data Lakehouse

Thank you for your interest in contributing to the Data Lakehouse project! This document provides guidelines and instructions for contributing.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Process](#development-process)
4. [Code Standards](#code-standards)
5. [Testing Requirements](#testing-requirements)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Review Process](#review-process)

---

## Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior
- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior
- Harassment, discrimination, or offensive comments
- Trolling or insulting/derogatory comments
- Public or private harassment
- Publishing others' private information
- Other conduct deemed inappropriate

---

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/your-username/data-lakehouse.git
cd data-lakehouse

# Add upstream remote
git remote add upstream https://github.com/original-org/data-lakehouse.git
```

### 2. Development Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

**requirements-dev.txt:**
```
# Testing
pytest>=7.2.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
pytest-asyncio>=0.20.0
pytest-benchmark>=4.0.0

# Code quality
black>=22.12.0
flake8>=6.0.0
mypy>=0.991
pylint>=2.15.0
isort>=5.11.0
bandit>=1.7.4

# Documentation
sphinx>=5.3.0
sphinx-rtd-theme>=1.1.0
sphinx-autodoc-typehints>=1.19.0

# Development tools
pre-commit>=2.21.0
ipython>=8.7.0
ipdb>=0.13.11
```

### 3. Install Spark Locally

```bash
# Download Spark
wget https://dlcdn.apache.org/spark/spark-3.4.0/spark-3.4.0-bin-hadoop3.tgz
tar -xzf spark-3.4.0-bin-hadoop3.tgz
export SPARK_HOME=$PWD/spark-3.4.0-bin-hadoop3
export PATH=$PATH:$SPARK_HOME/bin

# Download Delta Lake JARs
wget -P $SPARK_HOME/jars/ https://repo1.maven.org/maven2/io/delta/delta-core_2.12/2.3.0/delta-core_2.12-2.3.0.jar
```

---

## Development Process

### 1. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Development Workflow

```bash
# Make changes
vim src/lakehouse/your_module.py

# Run tests frequently
pytest tests/unit/test_your_module.py -v

# Check code quality
black src/lakehouse/
flake8 src/lakehouse/
mypy src/lakehouse/

# Commit changes
git add .
git commit -m "feat: add new feature description"
```

### 3. Commit Message Convention

Follow the Conventional Commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `build`: Build system changes
- `ci`: CI/CD changes

**Examples:**
```bash
git commit -m "feat(processor): add support for Iceberg tables"
git commit -m "fix(optimizer): resolve memory leak in compaction"
git commit -m "docs(api): update transformation examples"
git commit -m "test(quality): add edge cases for validation"
```

---

## Code Standards

### Python Style Guide

Follow PEP 8 with these additions:

```python
# File header
"""Module description.

This module provides functionality for...
"""

from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ExampleClass:
    """Class description.

    Attributes:
        attribute1: Description of attribute1.
        attribute2: Description of attribute2.
    """

    def __init__(self, param1: str, param2: Optional[int] = None) -> None:
        """Initialize ExampleClass.

        Args:
            param1: Description of param1.
            param2: Optional description of param2.

        Raises:
            ValueError: If param1 is empty.
        """
        if not param1:
            raise ValueError("param1 cannot be empty")

        self.param1 = param1
        self.param2 = param2 or 0

    def example_method(self, data: Dict[str, Any]) -> List[str]:
        """Process data and return results.

        Args:
            data: Input data dictionary.

        Returns:
            List of processed strings.

        Example:
            >>> obj = ExampleClass("test")
            >>> obj.example_method({"key": "value"})
            ['processed_value']
        """
        results = []
        for key, value in data.items():
            processed = self._process_item(key, value)
            results.append(processed)

        logger.info(f"Processed {len(results)} items")
        return results

    def _process_item(self, key: str, value: Any) -> str:
        """Private method for processing items."""
        return f"{key}_{value}"
```

### Code Quality Checks

#### Black Configuration
```toml
# pyproject.toml
[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310']
include = '\.pyi?$'
extend-exclude = '''
/(
    \.git
  | \.venv
  | build
  | dist
)/
'''
```

#### Flake8 Configuration
```ini
# .flake8
[flake8]
max-line-length = 100
exclude = .git,__pycache__,venv,build,dist
ignore = E203,W503,E501
per-file-ignores =
    __init__.py:F401
```

#### MyPy Configuration
```ini
# mypy.ini
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_any_unimported = False
no_implicit_optional = True
check_untyped_defs = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 22.12.0
    hooks:
      - id: black
        language_version: python3.8

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pycqa/isort
    rev: 5.11.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.991
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

---

## Testing Requirements

### Test Structure

```
tests/
├── unit/
│   ├── test_processor.py
│   ├── test_optimizer.py
│   └── test_quality.py
├── integration/
│   ├── test_pipeline.py
│   └── test_streaming.py
├── fixtures/
│   ├── sample_data.py
│   └── spark_fixture.py
└── conftest.py
```

### Writing Tests

```python
# tests/unit/test_processor.py
import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession
from lakehouse.processor import LakehouseProcessor


class TestLakehouseProcessor:
    """Test cases for LakehouseProcessor."""

    @pytest.fixture
    def spark(self):
        """Create SparkSession for testing."""
        return SparkSession.builder \
            .appName("test") \
            .master("local[2]") \
            .getOrCreate()

    @pytest.fixture
    def processor(self, spark):
        """Create processor instance."""
        return LakehouseProcessor(spark)

    def test_bronze_ingestion_success(self, processor, tmp_path):
        """Test successful bronze ingestion."""
        # Arrange
        source_path = tmp_path / "source.json"
        source_path.write_text('{"id": 1, "name": "test"}')
        bronze_path = str(tmp_path / "bronze")

        # Act
        processor.bronze_ingestion(
            source_path=str(source_path),
            bronze_path=bronze_path,
            source_name="test"
        )

        # Assert
        df = processor.spark.read.format("delta").load(bronze_path)
        assert df.count() == 1
        assert "ingestion_timestamp" in df.columns

    def test_bronze_ingestion_invalid_path(self, processor):
        """Test bronze ingestion with invalid path."""
        with pytest.raises(FileNotFoundError):
            processor.bronze_ingestion(
                source_path="/invalid/path",
                bronze_path="/bronze",
                source_name="test"
            )

    @patch('lakehouse.processor.DeltaTable')
    def test_merge_operation(self, mock_delta, processor):
        """Test merge operation with mocking."""
        # Arrange
        mock_table = Mock()
        mock_delta.forPath.return_value = mock_table

        # Act
        processor.merge_delta_tables(
            target_path="/target",
            source_df=Mock(),
            merge_condition="id = id"
        )

        # Assert
        mock_table.merge.assert_called_once()
```

### Test Coverage Requirements

- Minimum overall coverage: 80%
- Critical paths coverage: 95%
- New code coverage: 90%

Run coverage report:
```bash
pytest --cov=lakehouse --cov-report=html --cov-report=term
```

### Performance Testing

```python
# tests/performance/test_benchmark.py
import pytest
from lakehouse.optimizer import StorageOptimizer


@pytest.mark.benchmark(group="optimizer")
def test_compaction_performance(benchmark, spark, large_dataset):
    """Benchmark file compaction."""
    optimizer = StorageOptimizer(spark)

    result = benchmark(
        optimizer.compact_files,
        files=large_dataset,
        target_size_mb=128
    )

    assert result['files_created'] < result['files_processed']
```

---

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> Dict[str, Any]:
    """Short description of function.

    Longer description if needed, explaining the purpose
    and behavior of the function in more detail.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: Description of when this is raised.
        IOError: Description of when this is raised.

    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        {'status': 'success', 'value': 42}

    Note:
        Additional notes about the function.
    """
    pass
```

### API Documentation

Update API documentation when adding/modifying public APIs:

```bash
# Generate API docs
cd docs
make clean
make html

# View generated docs
open _build/html/index.html
```

### README Updates

Update README.md for:
- New features
- Configuration changes
- Breaking changes
- New dependencies

---

## Submitting Changes

### 1. Run Full Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v

# Check code quality
black --check src/
flake8 src/
mypy src/
pylint src/

# Security scan
bandit -r src/
```

### 2. Update Documentation

```bash
# Update CHANGELOG.md
echo "## [Unreleased]
### Added
- Your new feature" >> CHANGELOG.md

# Update version if needed
vim setup.py  # Update version number
```

### 3. Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create PR via GitHub UI or CLI
gh pr create \
  --title "feat: add new feature" \
  --body "Description of changes" \
  --base main
```

### Pull Request Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No new warnings

## Screenshots (if applicable)
Add screenshots here.

## Related Issues
Fixes #123
```

---

## Review Process

### Review Criteria

1. **Code Quality**
   - Follows style guide
   - No code smells
   - Proper error handling
   - Efficient algorithms

2. **Testing**
   - Adequate test coverage
   - Tests pass in CI
   - Edge cases covered

3. **Documentation**
   - Clear docstrings
   - Updated API docs
   - README updates if needed

4. **Performance**
   - No performance regressions
   - Efficient resource usage
   - Scalability considered

### Review Timeline

- Initial review: Within 2 business days
- Follow-up reviews: Within 1 business day
- Merge after 2 approvals

### Addressing Feedback

```bash
# Make requested changes
git add .
git commit -m "address review feedback"

# Or amend if preferred
git commit --amend

# Force push to your branch
git push --force-with-lease origin feature/your-feature-name
```

---

## Release Process

### Version Numbering

Follow Semantic Versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] Release notes prepared
- [ ] Tagged in git
- [ ] Published to PyPI

---

## Getting Help

### Resources
- [Documentation](https://docs.lakehouse.io)
- [Issue Tracker](https://github.com/org/lakehouse/issues)
- [Discussions](https://github.com/org/lakehouse/discussions)
- [Slack Channel](https://lakehouse.slack.com)

### Contact
- Email: lakehouse-dev@example.com
- Twitter: @lakehouse_io

Thank you for contributing to Data Lakehouse!