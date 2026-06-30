# Contributing to CRDT Collaboration

Thank you for your interest in contributing to the CRDT Collaboration project! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Process](#development-process)
4. [Code Style](#code-style)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Review Process](#review-process)

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Expected Behavior

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment, discriminatory language, or personal attacks
- Publishing private information without consent
- Trolling or insulting comments
- Any conduct that could reasonably be considered inappropriate

## Getting Started

### Prerequisites

1. Install Rust 1.70+ and Cargo
2. Fork the repository on GitHub
3. Clone your fork locally:

```bash
git clone https://github.com/yourusername/crdt-collaboration.git
cd crdt-collaboration
```

### Setting Up Development Environment

```bash
# Install development dependencies
cargo install cargo-watch cargo-edit cargo-audit

# Install pre-commit hooks
./scripts/install-hooks.sh

# Run initial build and tests
cargo build
cargo test

# Set up development database
./scripts/setup-dev.sh
```

### Understanding the Codebase

Key directories:
- `src/crdt.rs` - CRDT type implementations
- `src/document.rs` - Document management
- `src/server.rs` - WebSocket server
- `src/protocol.rs` - Wire protocol
- `tests/` - Test suites
- `docs/` - Documentation

## Development Process

### 1. Find or Create an Issue

Before starting work:

1. Check existing issues for similar work
2. If none exists, create a new issue describing:
   - The problem or feature
   - Your proposed solution
   - Any design considerations

### 2. Create a Feature Branch

```bash
# Create branch from main
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name

# Or for bugfixes
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or fixes

### 3. Make Your Changes

Follow these guidelines:

- Keep commits small and focused
- Write descriptive commit messages
- Add tests for new functionality
- Update documentation as needed

### 4. Commit Message Format

```
type(scope): brief description

Longer explanation of the change if necessary. Wrap at 72 characters.
Explain the problem this commit is solving.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Maintenance tasks

Example:
```
feat(crdt): add support for RGA list operations

Implements insert, delete, and merge operations for RGA lists.
This enables ordered list collaboration with proper conflict resolution.

Fixes #456
```

## Code Style

### Rust Style Guide

We follow the official Rust style guide with these additions:

```rust
// Use explicit imports
use std::collections::HashMap;
use crate::crdt::Operation;

// Document public APIs
/// Merges two CRDT instances deterministically.
///
/// # Arguments
/// * `other` - The other CRDT instance to merge
///
/// # Returns
/// A new merged CRDT instance
pub fn merge(&self, other: &Self) -> Self {
    // Implementation
}

// Use descriptive variable names
let operation_timestamp = Timestamp::now();
// Not: let ts = Timestamp::now();

// Error handling
fn process_operation(op: Operation) -> Result<(), Error> {
    // Prefer ? operator over unwrap()
    let validated = validate_operation(op)?;

    // Use match for complex error handling
    match apply_operation(validated) {
        Ok(result) => Ok(result),
        Err(e) if e.is_recoverable() => {
            recover_from_error(e)
        }
        Err(e) => Err(e),
    }
}
```

### Running Style Checks

```bash
# Format code
cargo fmt

# Check lints
cargo clippy -- -D warnings

# Check all
./scripts/check-style.sh
```

## Testing

### Test Organization

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests
├── fixtures/       # Test data and helpers
└── benchmarks/     # Performance benchmarks
```

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_merge() {
        // Arrange
        let op1 = create_test_operation();
        let op2 = create_test_operation();

        // Act
        let merged = op1.merge(&op2);

        // Assert
        assert_eq!(merged.timestamp, expected_timestamp);
        assert!(merged.is_valid());
    }

    #[tokio::test]
    async fn test_async_operation() {
        let doc = Document::new("test");
        doc.insert(0, "test").await.unwrap();
        assert_eq!(doc.content().await, "test");
    }
}
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test module
cargo test crdt::tests

# Run with output
cargo test -- --nocapture

# Run benchmarks
cargo bench

# Run with coverage
cargo tarpaulin --out Html
```

### Test Coverage

We aim for:
- 80%+ coverage for core CRDT algorithms
- 70%+ coverage for server code
- 60%+ overall project coverage

## Documentation

### Code Documentation

All public APIs must be documented:

```rust
/// Represents a collaborative document with CRDT-based conflict resolution.
///
/// # Example
///
/// ```rust
/// use crdt_collaboration::Document;
///
/// let mut doc = Document::new("doc-id", "replica-id");
/// doc.insert(0, "Hello").await?;
/// ```
pub struct Document {
    // ...
}
```

### Documentation Updates

When changing functionality:

1. Update inline documentation
2. Update relevant files in `docs/`
3. Update README if needed
4. Add examples for complex features

### Building Documentation

```bash
# Build and view docs
cargo doc --open

# Build with private items
cargo doc --document-private-items

# Check documentation
cargo doc --no-deps
```

## Submitting Changes

### Pre-submission Checklist

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Code is properly formatted (`cargo fmt`)
- [ ] Clippy passes (`cargo clippy`)
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main

### Creating a Pull Request

1. Push your branch:
```bash
git push origin feature/your-feature-name
```

2. Create PR on GitHub with:
   - Descriptive title
   - Summary of changes
   - Link to related issue
   - Screenshots/demos if applicable

3. PR template:
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
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guide
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## Review Process

### What to Expect

1. **Automated Checks**: CI will run tests and style checks
2. **Code Review**: Maintainers will review within 2-3 days
3. **Feedback**: Address review comments
4. **Approval**: Two approvals required for merge
5. **Merge**: Maintainer will merge when ready

### Review Guidelines

For reviewers:

- Focus on:
  - Correctness of CRDT algorithms
  - Test coverage
  - Documentation clarity
  - Performance implications
  - Security considerations

- Provide:
  - Constructive feedback
  - Specific suggestions
  - Links to resources when helpful

### Addressing Feedback

```bash
# Make requested changes
git add .
git commit -m "address review feedback"

# Or amend if small change
git commit --amend

# Push changes
git push origin feature/your-feature-name
```

## Performance Considerations

### Benchmarking

Before submitting performance-related changes:

```bash
# Run benchmarks
cargo bench -- --save-baseline before
# Make changes
cargo bench -- --baseline before
```

### Profiling

```bash
# CPU profiling
cargo build --release
perf record --call-graph=dwarf ./target/release/server
perf report

# Memory profiling
valgrind --tool=massif ./target/release/server
```

## Security

### Reporting Security Issues

DO NOT open public issues for security vulnerabilities. Instead:

1. Email security@crdt-collaboration.org
2. Include:
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Security Review

All PRs are reviewed for:
- Input validation
- Memory safety
- Authentication/authorization
- Data privacy
- Denial of service potential

## Community

### Getting Help

- GitHub Discussions for questions
- Discord server for real-time chat
- Stack Overflow tag: `crdt-collaboration`

### Becoming a Maintainer

Active contributors may be invited to become maintainers. Criteria:

- Consistent high-quality contributions
- Good understanding of CRDT theory
- Helpful in community discussions
- Commitment to project values

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT/Apache-2.0).

## Acknowledgments

Thank you to all contributors who help make this project better!