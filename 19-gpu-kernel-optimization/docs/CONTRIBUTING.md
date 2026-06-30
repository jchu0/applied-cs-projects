# Contributing to GPU GEMM Optimization

Thank you for your interest in contributing to the GPU GEMM Optimization project! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Code Style](#code-style)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Issue Guidelines](#issue-guidelines)

## Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, gender identity, sexual orientation, disability, personal appearance, race, ethnicity, age, religion, or nationality.

### Expected Behavior
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully
- Respect differing viewpoints

### Unacceptable Behavior
- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information
- Any conduct that would be inappropriate in a professional setting

## Getting Started

### Prerequisites
- Rust 1.70.0 or later
- Git
- A GitHub account
- Familiarity with GEMM operations and linear algebra (helpful but not required)

### First-Time Contributors
Look for issues labeled with:
- `good first issue` - Simple issues perfect for beginners
- `help wanted` - Issues where we need community help
- `documentation` - Documentation improvements
- `testing` - Test coverage improvements

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/gpu-gemm-optimization.git
cd gpu-gemm-optimization

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/gpu-gemm-optimization.git
```

### 2. Create a Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create a feature branch
git checkout -b feature/your-feature-name
# Or for bugs:
git checkout -b fix/bug-description
```

### 3. Set Up Development Environment

```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install development tools
rustup component add rustfmt clippy

# Install cargo-watch for auto-recompilation
cargo install cargo-watch

# Install benchmarking tools
cargo install cargo-criterion

# Build the project
cargo build

# Run tests to verify setup
cargo test
```

### 4. Install Pre-commit Hooks

```bash
# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test --quiet
EOF

chmod +x .git/hooks/pre-commit
```

## How to Contribute

### Types of Contributions

#### 1. Bug Fixes
- Reproduce the bug
- Write a failing test
- Implement the fix
- Ensure all tests pass

#### 2. New Features
- Discuss the feature in an issue first
- Write comprehensive tests
- Update documentation
- Add examples if applicable

#### 3. Performance Improvements
- Benchmark before and after
- Document the optimization technique
- Ensure correctness is maintained
- Update performance tests

#### 4. Documentation
- Fix typos and improve clarity
- Add examples and use cases
- Update API documentation
- Improve README and guides

#### 5. Tests
- Increase test coverage
- Add edge case tests
- Improve test performance
- Add property-based tests

### Contribution Workflow

1. **Check existing issues and PRs**
2. **Create or claim an issue**
3. **Develop your contribution**
4. **Test thoroughly**
5. **Update documentation**
6. **Submit pull request**
7. **Address review feedback**

## Code Style

### Rust Style Guide

We follow the official Rust style guide with some project-specific conventions:

#### Formatting
```bash
# Format your code
cargo fmt

# Check formatting
cargo fmt --check
```

#### Linting
```bash
# Run clippy
cargo clippy -- -D warnings

# Fix clippy suggestions
cargo clippy --fix
```

#### Naming Conventions
```rust
// Modules: snake_case
mod matrix_operations;

// Types: PascalCase
struct GemmKernel;
enum OptimizationStrategy;

// Functions: snake_case
fn compute_gemm() {}

// Constants: SCREAMING_SNAKE_CASE
const MAX_TILE_SIZE: usize = 256;

// Variables: snake_case
let tile_config = TileConfig::default();
```

#### Documentation
```rust
/// Brief description of the function.
///
/// # Arguments
///
/// * `a` - First input matrix
/// * `b` - Second input matrix
///
/// # Returns
///
/// Returns the result matrix or an error.
///
/// # Examples
///
/// ```
/// let a = Matrix::random(10, 10);
/// let b = Matrix::random(10, 10);
/// let c = gemm(&a, &b)?;
/// ```
///
/// # Errors
///
/// Returns `GemmError` if matrices have incompatible dimensions.
pub fn gemm(a: &Matrix, b: &Matrix) -> Result<Matrix, GemmError> {
    // Implementation
}
```

### Code Organization

```
src/
├── lib.rs           # Public API exports
├── matrix.rs        # Matrix module
├── gemm.rs         # GEMM kernels
├── autotuner.rs    # Autotuning system
├── metrics.rs      # Performance metrics
└── utils/          # Utility modules
    ├── mod.rs
    └── ...
```

### Error Handling

```rust
// Use Result types for fallible operations
pub fn risky_operation() -> Result<Output, Error> {
    // ...
}

// Use custom error types
#[derive(Debug, thiserror::Error)]
pub enum GemmError {
    #[error("Matrix dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Numerical overflow detected")]
    NumericOverflow,
}

// Propagate errors with ?
fn process() -> Result<(), Box<dyn Error>> {
    let result = risky_operation()?;
    Ok(())
}
```

## Testing

### Test Organization

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── performance/    # Performance benchmarks
└── fixtures/       # Test data and helpers
```

### Writing Tests

#### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let mat = Matrix::zeros(10, 10);
        assert_eq!(mat.rows(), 10);
        assert_eq!(mat.cols(), 10);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_invalid_dimensions() {
        let a = Matrix::zeros(3, 4);
        let b = Matrix::zeros(5, 3);
        let _ = gemm(&a, &b);
    }
}
```

#### Integration Tests
```rust
// tests/integration/test_full_workflow.rs
use gpu_gemm_optimization::*;

#[test]
fn test_complete_optimization_workflow() {
    // Test the entire pipeline
    let a = Matrix::random(256, 256);
    let b = Matrix::random(256, 256);

    let mut tuner = Autotuner::new(AutotuneConfig::default()).unwrap();
    let config = tuner.tune(&a, &b).unwrap();

    let kernel = GemmKernel::new(config);
    let result = kernel.execute(&a, &b).unwrap();

    assert!(result.is_valid());
}
```

#### Performance Tests
```rust
// benches/gemm_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_gemm(c: &mut Criterion) {
    let a = Matrix::random(512, 512);
    let b = Matrix::random(512, 512);

    c.bench_function("gemm_512x512", |bench| {
        bench.iter(|| {
            gemm(black_box(&a), black_box(&b))
        });
    });
}

criterion_group!(benches, benchmark_gemm);
criterion_main!(benches);
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_matrix_creation

# Run tests with output
cargo test -- --nocapture

# Run tests in parallel
cargo test -- --test-threads=4

# Run benchmarks
cargo bench

# Run with coverage (requires cargo-tarpaulin)
cargo tarpaulin --out Html
```

### Test Coverage

We aim for:
- **Unit tests**: 80%+ coverage
- **Integration tests**: All major workflows
- **Edge cases**: Comprehensive coverage
- **Performance**: Regression tests for all optimizations

## Documentation

### Documentation Standards

#### Code Documentation
- All public APIs must be documented
- Include examples for complex functions
- Document performance characteristics
- Explain algorithm choices

#### README Updates
- Keep feature list current
- Update installation instructions
- Maintain compatibility information
- Include benchmark results

#### Architecture Documentation
- Document design decisions
- Explain optimization strategies
- Include diagrams where helpful
- Keep it up-to-date with code

### Building Documentation

```bash
# Generate API documentation
cargo doc --no-deps --open

# Generate documentation with private items
cargo doc --no-deps --document-private-items

# Check documentation examples
cargo test --doc
```

## Pull Request Process

### Before Submitting

1. **Update your branch**
```bash
git fetch upstream
git rebase upstream/main
```

2. **Run full test suite**
```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo bench --no-run  # Ensure benchmarks compile
```

3. **Update documentation**
- Add/update rustdoc comments
- Update README if needed
- Add to CHANGELOG.md

4. **Commit with meaningful messages**
```bash
# Good commit messages:
git commit -m "feat: Add Bayesian optimization to autotuner"
git commit -m "fix: Correct tile boundary handling in GEMM kernel"
git commit -m "perf: Optimize cache prefetch distance calculation"
git commit -m "docs: Add examples for custom memory layouts"

# Use conventional commits:
# feat: New feature
# fix: Bug fix
# perf: Performance improvement
# docs: Documentation
# test: Tests
# refactor: Code restructuring
# style: Formatting changes
```

### Submitting the PR

1. **Push to your fork**
```bash
git push origin feature/your-feature-name
```

2. **Create pull request**
- Use a clear, descriptive title
- Reference related issues
- Describe what changes you made
- Explain why the changes are needed
- Include benchmark results if applicable

3. **PR Template**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Benchmarks run successfully

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No new warnings
```

### Review Process

1. **Automated checks** - CI will run tests and linting
2. **Code review** - Maintainers will review the code
3. **Feedback** - Address any requested changes
4. **Approval** - Once approved, PR will be merged

### After Merge

- Delete your feature branch
- Update your fork
- Celebrate your contribution!

## Issue Guidelines

### Reporting Bugs

```markdown
## Bug Description
Clear description of the bug

## To Reproduce
1. Step one
2. Step two
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## System Information
- OS: [e.g., Ubuntu 22.04]
- Rust version: [e.g., 1.70.0]
- Project version: [e.g., 0.1.0]

## Additional Context
Any other relevant information
```

### Requesting Features

```markdown
## Feature Description
What feature would you like to see?

## Use Case
Why is this feature needed?

## Proposed Implementation
How might this be implemented?

## Alternatives Considered
What alternatives have you considered?
```

### Performance Issues

```markdown
## Performance Issue
Description of performance problem

## Benchmarks
Current performance metrics

## Expected Performance
Target performance metrics

## Profiling Data
Any profiling information

## System Specifications
Hardware and software details
```

## Recognition

### Contributors
All contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

### Types of Recognition
- **Code contributions**: Listed as contributors
- **Bug reports**: Acknowledged in fix commits
- **Feature ideas**: Credited in implementations
- **Documentation**: Listed as documentation contributors

## Questions?

### Getting Help
- **Documentation**: Check docs/ directory
- **Examples**: See examples/ directory
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions

### Contact
- Open an issue for bugs
- Start a discussion for questions
- Email maintainers for sensitive matters

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

## Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort in improving GPU GEMM Optimization!