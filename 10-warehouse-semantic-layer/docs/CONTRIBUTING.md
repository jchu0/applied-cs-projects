# Contributing to Warehouse Semantic Layer

Thank you for your interest in contributing to the Warehouse Semantic Layer project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a friendly, safe, and welcoming environment for all contributors, regardless of experience level, gender identity and expression, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, religion, nationality, or other similar characteristics.

### Expected Behavior

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully
- Prioritize the project's best interests

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or insults
- Publishing others' private information
- Any conduct that could reasonably be considered inappropriate

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- A GitHub account
- Familiarity with data warehouses and SQL

### First-Time Contributors

Look for issues labeled with:
- `good first issue` - Simple issues perfect for beginners
- `help wanted` - Issues where we need community help
- `documentation` - Documentation improvements

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates.

**Bug Report Template:**

```markdown
### Description
Clear description of the bug

### Steps to Reproduce
1. Configure semantic layer with...
2. Execute query...
3. See error...

### Expected Behavior
What should happen

### Actual Behavior
What actually happens

### Environment
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.10.0]
- Semantic Layer version: [e.g., 1.2.0]
- Data Warehouse: [e.g., Snowflake]

### Additional Context
Any other relevant information
```

### Suggesting Features

**Feature Request Template:**

```markdown
### Problem Statement
What problem does this feature solve?

### Proposed Solution
How would this feature work?

### Alternatives Considered
Other approaches you've thought about

### Additional Context
Use cases, examples, or mockups
```

### Contributing Code

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Write/update tests**
5. **Update documentation**
6. **Submit a pull request**

## Development Setup

### 1. Fork and Clone

```bash
# Fork via GitHub UI, then:
git clone https://github.com/YOUR_USERNAME/warehouse-semantic-layer.git
cd warehouse-semantic-layer
git remote add upstream https://github.com/original/warehouse-semantic-layer.git
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install project dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 4. Set Up Test Database

```bash
# Start test containers
docker-compose -f docker-compose.test.yml up -d

# Run database migrations
python manage.py migrate
```

### 5. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=semantic_layer --cov-report=term-missing

# Run specific tests
pytest tests/test_query_engine.py::TestSemanticQueryEngine
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good example
from typing import List, Optional

from semantic_layer.models import MetricDefinition


class MetricProcessor:
    """Process metric definitions for optimization.

    This class handles the optimization of metric queries
    by analyzing dependencies and suggesting indexes.
    """

    def __init__(self, metrics: List[MetricDefinition]) -> None:
        """Initialize the processor.

        Args:
            metrics: List of metric definitions to process
        """
        self.metrics = metrics
        self._cache = {}

    def optimize_query(
        self,
        metric_name: str,
        dimensions: Optional[List[str]] = None
    ) -> str:
        """Optimize a metric query.

        Args:
            metric_name: Name of the metric to optimize
            dimensions: Optional list of dimensions

        Returns:
            Optimized SQL query string

        Raises:
            ValueError: If metric not found
        """
        if metric_name not in self._cache:
            self._cache[metric_name] = self._build_optimized_query(
                metric_name, dimensions
            )
        return self._cache[metric_name]
```

### Code Quality Tools

```bash
# Format code
black semantic_layer tests

# Sort imports
isort semantic_layer tests

# Type checking
mypy semantic_layer

# Linting
flake8 semantic_layer tests
pylint semantic_layer

# Security scanning
bandit -r semantic_layer
```

### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--extend-ignore=E203']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Unit tests for individual components
│   ├── test_models.py
│   ├── test_query_engine.py
│   └── test_api.py
├── integration/       # Integration tests
│   ├── test_warehouse_connectors.py
│   └── test_end_to_end.py
├── fixtures/         # Test fixtures and sample data
│   ├── metrics.yaml
│   └── sample_data.sql
└── conftest.py      # Shared pytest fixtures
```

### Writing Tests

```python
# Good test example
import pytest
from unittest.mock import Mock, patch

from semantic_layer.query_engine import SemanticQueryEngine


class TestSemanticQueryEngine:
    """Test the semantic query engine."""

    @pytest.fixture
    def engine(self, metric_catalog):
        """Create an engine instance."""
        return SemanticQueryEngine(metric_catalog)

    def test_generate_sql_with_single_metric(self, engine):
        """Test SQL generation for a single metric."""
        # Arrange
        query = Mock(
            metrics=["revenue"],
            dimensions=["region"],
            time_grain="month",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        # Act
        sql = engine.generate_sql(query)

        # Assert
        assert "SELECT" in sql
        assert "SUM(amount) as revenue" in sql
        assert "GROUP BY" in sql
        assert "region" in sql

    @patch('semantic_layer.query_engine.logger')
    def test_error_handling(self, mock_logger, engine):
        """Test error handling and logging."""
        # Test invalid metric
        with pytest.raises(ValueError, match="Metric not found"):
            engine.generate_sql(Mock(metrics=["invalid"]))

        # Verify logging
        mock_logger.error.assert_called()
```

### Test Coverage Requirements

- Minimum 80% code coverage for new code
- 100% coverage for critical paths (query engine, API endpoints)
- Integration tests for all warehouse types

## Documentation

### Documentation Standards

All code should be well-documented:

```python
def calculate_metric(
    metric_name: str,
    filters: Optional[List[Dict[str, Any]]] = None,
    time_grain: str = "day"
) -> float:
    """Calculate a metric value with optional filters.

    This function calculates the value of a metric by generating
    and executing the appropriate SQL query against the warehouse.

    Args:
        metric_name: Name of the metric to calculate
        filters: Optional list of filter dictionaries with keys:
            - field: Field name to filter on
            - operator: Comparison operator (=, !=, <, >, in, etc.)
            - value: Value to compare against
        time_grain: Time granularity for aggregation (day, week, month)

    Returns:
        Calculated metric value as a float

    Raises:
        ValueError: If metric_name is not found in catalog
        ConnectionError: If warehouse connection fails
        QueryTimeout: If query exceeds timeout limit

    Example:
        >>> calculate_metric(
        ...     "revenue",
        ...     filters=[{"field": "region", "operator": "=", "value": "US"}],
        ...     time_grain="month"
        ... )
        1500000.00

    Note:
        Results are cached for 1 hour by default. Use the cache_ttl
        parameter to override this behavior.
    """
    # Implementation here
```

### Documentation Types

1. **API Documentation**: Update `docs/API.md` for new endpoints
2. **Architecture Documentation**: Update `docs/ARCHITECTURE.md` for design changes
3. **User Guides**: Add examples and tutorials in `docs/guides/`
4. **Changelog**: Update `CHANGELOG.md` with notable changes

## Pull Request Process

### Before Submitting

1. **Update your fork**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes and commit**
   ```bash
   git add .
   git commit -m "feat: add support for window functions in metrics"
   ```

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `test:` Test additions or changes
- `chore:` Maintenance tasks

Examples:
```
feat: add BigQuery warehouse connector
fix: resolve SQL injection vulnerability in filter parsing
docs: update API documentation for v2 endpoints
perf: optimize metric catalog search with indexing
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Dependent changes merged

## Testing
Description of tests performed

## Screenshots (if applicable)
Add screenshots for UI changes

## Related Issues
Closes #123
```

### Review Process

1. **Automated Checks**: CI/CD runs tests, linting, and security scans
2. **Code Review**: At least one maintainer reviews the code
3. **Testing**: Reviewer may test locally
4. **Feedback**: Address review comments
5. **Approval**: Once approved, maintainer merges

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General discussions and questions
- **Slack**: [Join our Slack](https://semantic-layer.slack.com)
- **Email**: semantic-layer@example.com

### Getting Help

- Check the [documentation](https://docs.semantic-layer.io)
- Search existing issues and discussions
- Ask in the Slack channel
- Create a new issue with the `question` label

### Recognition

We value all contributions! Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project website

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Build documentation
5. Tag release: `git tag -a v1.2.0 -m "Release v1.2.0"`
6. Push tag: `git push upstream v1.2.0`
7. Create GitHub release
8. Publish to PyPI

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

## Questions?

Feel free to reach out if you have any questions! We're here to help make your contribution experience as smooth as possible.

Thank you for contributing to the Warehouse Semantic Layer! 🎉