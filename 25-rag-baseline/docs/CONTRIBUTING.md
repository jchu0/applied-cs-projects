# Contributing to RAG Baseline

Thank you for your interest in contributing to RAG Baseline! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

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
- Harassment of any kind
- Trolling, insulting/derogatory comments
- Public or private harassment
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. Python 3.9 or higher installed
2. Git configured with your GitHub account
3. A fork of the RAG Baseline repository
4. Basic understanding of RAG systems and Python

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/rag-baseline.git
cd rag-baseline

# Add upstream remote
git remote add upstream https://github.com/original-org/rag-baseline.git

# Verify remotes
git remote -v
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **Bug Fixes**: Fix issues reported in GitHub Issues
2. **Features**: Implement new features or enhance existing ones
3. **Documentation**: Improve or add documentation
4. **Tests**: Add or improve test coverage
5. **Performance**: Optimize code for better performance
6. **Refactoring**: Improve code structure and readability

### Finding Issues to Work On

Look for issues labeled:
- `good-first-issue`: Perfect for newcomers
- `help-wanted`: Community help needed
- `enhancement`: Feature improvements
- `bug`: Bug fixes needed
- `documentation`: Documentation improvements

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

# Set up test environment
cp .env.test.example .env.test
```

### Development Dependencies

```bash
# Install all development dependencies
pip install -e ".[dev,test,docs]"

# Core development tools
# - pytest: Testing framework
# - black: Code formatting
# - flake8: Linting
# - mypy: Type checking
# - pre-commit: Git hooks
# - sphinx: Documentation
```

### Running the Development Server

```bash
# Start the development server with auto-reload
uvicorn ragbaseline.api:app --reload --host 0.0.0.0 --port 8000

# Or use the development script
python scripts/dev_server.py
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good: Descriptive variable names
embedding_dimension = 1536
chunk_overlap_tokens = 50

# Bad: Single letter or unclear names
d = 1536
o = 50

# Good: Type hints for all functions
def process_document(
    content: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[Chunk]:
    """Process document into chunks.

    Args:
        content: Document content to process
        chunk_size: Size of each chunk in tokens
        overlap: Number of overlapping tokens

    Returns:
        List of document chunks
    """
    pass

# Good: Docstrings for all public functions/classes
class RAGPipeline:
    """Main RAG pipeline for query processing.

    This class orchestrates the retrieval and generation
    process for answering queries using RAG.

    Attributes:
        retriever: Document retriever instance
        generator: LLM generator instance
        config: Pipeline configuration
    """
```

### Code Formatting

```bash
# Format code with black
black ragbaseline/ tests/

# Check formatting without modifying
black --check ragbaseline/ tests/

# Sort imports
isort ragbaseline/ tests/

# Run linting
flake8 ragbaseline/ tests/

# Type checking
mypy ragbaseline/
```

### Naming Conventions

```python
# Classes: PascalCase
class DocumentProcessor:
    pass

# Functions/variables: snake_case
def process_query(query_text: str) -> str:
    max_tokens = 500

# Constants: UPPER_SNAKE_CASE
DEFAULT_CHUNK_SIZE = 512
MAX_RETRIEVAL_DOCUMENTS = 10

# Private methods: leading underscore
def _internal_function():
    pass

# Module-level private: leading underscore
_INTERNAL_CONSTANT = 42
```

### Error Handling

```python
# Good: Specific exception handling
try:
    result = await vectorstore.search(query)
except VectorStoreConnectionError as e:
    logger.error(f"Failed to connect to vector store: {e}")
    raise ServiceUnavailableError("Search service temporarily unavailable")
except VectorStoreTimeoutError as e:
    logger.warning(f"Search timeout: {e}")
    return fallback_search(query)

# Good: Custom exceptions
class RAGException(Exception):
    """Base exception for RAG operations."""
    pass

class RetrievalException(RAGException):
    """Exception in document retrieval."""
    pass
```

## Testing Guidelines

### Writing Tests

```python
# tests/test_retrieval.py
import pytest
from unittest.mock import Mock, AsyncMock

class TestVectorRetriever:
    """Tests for VectorRetriever class."""

    @pytest.fixture
    def mock_vectorstore(self):
        """Create mock vector store."""
        store = Mock()
        store.search = AsyncMock(return_value=[])
        return store

    @pytest.mark.asyncio
    async def test_retrieve_with_filter(self, mock_vectorstore):
        """Test retrieval with metadata filter."""
        # Arrange
        retriever = VectorRetriever(mock_vectorstore)
        query = "test query"
        filter = {"category": "science"}

        # Act
        results = await retriever.retrieve(query, filter=filter)

        # Assert
        mock_vectorstore.search.assert_called_once_with(
            query, filter=filter
        )
        assert isinstance(results, list)
```

### Test Organization

```
tests/
  unit/           # Unit tests for individual components
    test_embeddings.py
    test_chunking.py
    test_retrieval.py
  integration/    # Integration tests
    test_pipeline.py
    test_api.py
  e2e/           # End-to-end tests
    test_full_flow.py
  fixtures/      # Test fixtures and data
    documents/
    responses/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ragbaseline --cov-report=html

# Run specific test file
pytest tests/unit/test_retrieval.py

# Run specific test
pytest tests/unit/test_retrieval.py::TestVectorRetriever::test_retrieve_with_filter

# Run tests in parallel
pytest -n auto

# Run with verbose output
pytest -v

# Run only marked tests
pytest -m "not slow"
```

### Test Coverage Requirements

- Minimum 60% overall coverage
- 80% coverage for new code
- 100% coverage for critical paths (retrieval, generation)

## Documentation

### Docstring Format

We use Google-style docstrings:

```python
def enhance_query(
    query: str,
    context: Optional[str] = None,
    max_length: int = 100
) -> str:
    """Enhance user query with additional context.

    This function takes a user query and optionally adds
    context to improve retrieval results.

    Args:
        query: The original user query
        context: Optional context to add to the query
        max_length: Maximum length of enhanced query

    Returns:
        The enhanced query string

    Raises:
        ValueError: If query is empty
        QueryTooLongError: If enhanced query exceeds max_length

    Example:
        >>> enhance_query("What is AI?", context="machine learning")
        "What is AI? Context: machine learning"
    """
    if not query:
        raise ValueError("Query cannot be empty")

    enhanced = query
    if context:
        enhanced = f"{query} Context: {context}"

    if len(enhanced) > max_length:
        raise QueryTooLongError(f"Query exceeds {max_length} characters")

    return enhanced
```

### Building Documentation

```bash
# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html

# Check for documentation issues
sphinx-build -W -b html . _build/html
```

### Documentation Structure

```
docs/
  index.rst         # Main documentation index
  installation.rst  # Installation guide
  quickstart.rst   # Quick start guide
  api/             # API documentation
    index.rst
    retrieval.rst
    generation.rst
  tutorials/       # Step-by-step tutorials
    basic_rag.rst
    advanced.rst
  reference/       # API reference (auto-generated)
```

## Pull Request Process

### Before Creating a PR

1. **Sync with upstream**
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
# Make your changes
git add .
git commit -m "feat: add new retrieval strategy"
```

### Commit Message Format

We follow Conventional Commits:

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
- `test`: Test changes
- `chore`: Build process or auxiliary tool changes

**Examples:**
```bash
feat(retrieval): add MMR diversity algorithm

fix(api): handle empty query responses correctly

docs(readme): update installation instructions

perf(embeddings): optimize batch processing
```

### Creating the PR

1. **Push to your fork**
```bash
git push origin feature/your-feature-name
```

2. **Open PR on GitHub**
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Link related issues
   - Add appropriate labels

3. **PR Template**
```markdown
## Description
Brief description of changes

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

## Related Issues
Fixes #123
```

### Review Process

1. **Automated checks** must pass:
   - CI/CD pipeline
   - Code formatting
   - Tests
   - Coverage requirements

2. **Code review** by maintainers:
   - At least one approval required
   - Address all feedback
   - Resolve all conversations

3. **Final merge**:
   - Squash and merge for features
   - Rebase and merge for fixes

## Issue Guidelines

### Creating Issues

Use appropriate templates:

**Bug Report:**
```markdown
## Description
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
- Python: [e.g., 3.9.5]
- RAG Baseline version: [e.g., 1.2.0]

## Additional Context
Any other relevant information
```

**Feature Request:**
```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed?

## Proposed Solution
How should it work?

## Alternatives Considered
Other approaches considered

## Additional Context
Any other relevant information
```

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good-first-issue`: Good for newcomers
- `help-wanted`: Extra attention needed
- `question`: Further information requested
- `wontfix`: Will not be worked on

## Community

### Weekly Meetings

- **Community Call**: Thursdays at 3 PM UTC
- **Agenda**: Posted in Discord #meetings channel
- **Recording**: Available on YouTube

### Recognition

Contributors are recognized through:
- CONTRIBUTORS.md file
- GitHub contributor badge
- Community spotlight in newsletters

## Release Process

### Version Numbering

We follow Semantic Versioning (SemVer):
- MAJOR.MINOR.PATCH
- Example: 2.1.3

### Release Checklist

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Build and test Docker images
5. Create release PR
6. Tag release after merge
7. Publish to PyPI
8. Update documentation

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (Apache 2.0).

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search closed issues
3. Ask in Discord #help channel
4. Open a discussion thread

Thank you for contributing to RAG Baseline! 🎉