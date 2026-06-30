# Contributing to Long-Context Attention

Thank you for your interest in contributing to the Long-Context Attention project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We pledge to:

- Be respectful and considerate in all interactions
- Welcome contributors from all backgrounds
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Expected Behavior

- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept feedback gracefully
- Focus on collaboration over competition
- Help others learn and grow

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information
- Conduct that would be inappropriate in a professional setting

## Getting Started

### Prerequisites

- Rust 1.70.0 or higher
- Git
- GitHub account
- Familiarity with attention mechanisms and transformer architectures

### Setting Up Your Environment

1. **Fork the Repository**
   ```bash
   # Fork via GitHub UI, then clone your fork
   git clone https://github.com/YOUR_USERNAME/long-context-attention.git
   cd long-context-attention
   ```

2. **Add Upstream Remote**
   ```bash
   git remote add upstream https://github.com/original/long-context-attention.git
   git fetch upstream
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

## Development Setup

### Build Environment

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install development tools
cargo install cargo-watch cargo-tarpaulin cargo-audit

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Building the Project

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Build with all features
cargo build --all-features

# Build documentation
cargo doc --open
```

### Running Tests

```bash
# Run all tests
cargo test

# Run with coverage
cargo tarpaulin --out Html

# Run specific test
cargo test test_flash_attention

# Run benchmarks
cargo bench

# Run integration tests
cargo test --test integration_tests
```

## How to Contribute

### Types of Contributions

#### 1. Bug Reports

- Check existing issues first
- Provide a clear description
- Include reproduction steps
- Share system information
- Attach relevant logs

**Bug Report Template:**
```markdown
### Description
Brief description of the bug

### Reproduction Steps
1. Step one
2. Step two
3. ...

### Expected Behavior
What should happen

### Actual Behavior
What actually happens

### Environment
- OS: [e.g., Ubuntu 22.04]
- Rust version: [e.g., 1.70.0]
- GPU: [e.g., NVIDIA RTX 3090]
- CUDA version: [if applicable]

### Logs
```
Relevant error messages or logs
```
```

#### 2. Feature Requests

- Explain the motivation
- Describe the proposed solution
- Consider alternatives
- Discuss implementation approach

**Feature Request Template:**
```markdown
### Motivation
Why is this feature needed?

### Proposed Solution
Detailed description of the feature

### Alternatives Considered
Other approaches that were considered

### Implementation Notes
Technical details or considerations
```

#### 3. Code Contributions

- Fix bugs
- Implement new features
- Improve performance
- Refactor code
- Add tests

#### 4. Documentation

- Improve existing documentation
- Add examples
- Fix typos
- Translate documentation
- Write tutorials

## Coding Standards

### Rust Style Guide

Follow the official Rust style guide and use `rustfmt`:

```bash
# Format code
cargo fmt

# Check formatting
cargo fmt -- --check
```

### Code Organization

```
src/
├── lib.rs           # Library entry point
├── attention.rs     # Core attention interface
├── flash/          # Flash attention module
│   ├── mod.rs
│   ├── forward.rs
│   └── backward.rs
├── sparse/         # Sparse attention module
│   ├── mod.rs
│   └── patterns.rs
└── utils/          # Utility functions
    ├── mod.rs
    └── memory.rs
```

### Naming Conventions

```rust
// Modules: snake_case
mod attention_mechanism;

// Types: PascalCase
struct FlashAttention;
enum AttentionType;
trait AttentionMechanism;

// Functions and variables: snake_case
fn compute_attention();
let seq_length = 1024;

// Constants: SCREAMING_SNAKE_CASE
const MAX_SEQ_LEN: usize = 8192;
```

### Documentation Style

```rust
/// Brief description of the function.
///
/// Detailed explanation of what the function does,
/// including any important algorithmic details.
///
/// # Arguments
///
/// * `q` - Query tensor of shape [batch, heads, seq_len, head_dim]
/// * `k` - Key tensor of shape [batch, heads, seq_len, head_dim]
/// * `v` - Value tensor of shape [batch, heads, seq_len, head_dim]
///
/// # Returns
///
/// Attention output tensor of shape [batch, heads, seq_len, head_dim]
///
/// # Examples
///
/// ```rust
/// let output = flash_attention.forward(&q, &k, &v);
/// ```
///
/// # Panics
///
/// Panics if input tensors have incompatible shapes.
pub fn forward(&mut self, q: &Array4<f32>, k: &Array4<f32>, v: &Array4<f32>) -> Array4<f32> {
    // Implementation
}
```

### Error Handling

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AttentionError {
    #[error("Invalid input shape: expected {expected}, got {actual}")]
    InvalidShape { expected: String, actual: String },

    #[error("Out of memory: required {required} bytes, available {available} bytes")]
    OutOfMemory { required: usize, available: usize },

    #[error("CUDA error: {0}")]
    CudaError(String),
}

// Use Result type for fallible operations
pub fn process() -> Result<Array4<f32>, AttentionError> {
    // Implementation
}
```

## Testing Guidelines

### Test Structure

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Arrange
        let config = create_test_config();

        // Act
        let result = function_under_test(config);

        // Assert
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "error message")]
    fn test_error_condition() {
        // Test that should panic
    }
}
```

### Test Coverage Requirements

- Minimum 60% code coverage
- All public APIs must have tests
- Critical paths require >80% coverage
- Performance-critical code needs benchmarks

### Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_attention_properties(
        seq_len in 1..1024usize,
        batch_size in 1..8usize,
    ) {
        // Test invariants hold for random inputs
        let output = compute_attention(seq_len, batch_size);
        prop_assert!(output.iter().all(|x| x.is_finite()));
    }
}
```

### Benchmark Tests

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_flash_attention(c: &mut Criterion) {
    let q = create_test_tensor(1024, 8, 64);

    c.bench_function("flash_attention_1k", |b| {
        b.iter(|| {
            flash_attention.forward(black_box(&q), black_box(&k), black_box(&v))
        });
    });
}

criterion_group!(benches, bench_flash_attention);
criterion_main!(benches);
```

## Documentation

### Types of Documentation

1. **API Documentation**: Doc comments on all public items
2. **Architecture Docs**: High-level design documents
3. **User Guides**: How-to guides and tutorials
4. **Examples**: Working code examples

### Documentation Standards

- Write clear, concise descriptions
- Include examples for complex APIs
- Document all public items
- Keep documentation up-to-date with code changes

### Building Documentation

```bash
# Build and open documentation
cargo doc --open

# Build with private items
cargo doc --document-private-items

# Check documentation links
cargo doc --no-deps
```

## Pull Request Process

### Before Submitting

1. **Update from upstream**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**
   ```bash
   cargo test
   cargo fmt -- --check
   cargo clippy -- -D warnings
   ```

3. **Update documentation**
   - Add/update relevant documentation
   - Update CHANGELOG.md
   - Update README if needed

### PR Guidelines

#### Title Format
```
[Component] Brief description

Examples:
[Flash] Add support for dynamic block sizes
[Sparse] Fix memory leak in pattern generation
[Docs] Update installation instructions
```

#### Description Template
```markdown
## Summary
Brief description of changes

## Motivation
Why are these changes needed?

## Changes
- Change 1
- Change 2

## Testing
How were these changes tested?

## Checklist
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Code follows style guidelines
```

### Review Process

1. **Automated Checks**: CI runs tests, linting, and coverage
2. **Code Review**: At least one maintainer reviews
3. **Discussion**: Address feedback and iterate
4. **Approval**: Maintainer approves
5. **Merge**: Squash and merge to main

### After Merge

- Delete your feature branch
- Update your local main branch
- Celebrate your contribution! 🎉

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General discussions and questions
- **Discord**: Real-time chat (link in README)
- **Email**: maintainers@example.com

### Getting Help

- Check documentation first
- Search existing issues
- Ask in GitHub Discussions
- Join Discord for real-time help

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Annual contributor report

## Advanced Contributing

### Performance Optimization

When optimizing performance:

1. **Profile First**: Identify bottlenecks
2. **Measure Impact**: Benchmark before and after
3. **Document Changes**: Explain optimizations
4. **Consider Trade-offs**: Memory vs speed

### Adding New Attention Mechanisms

1. **Design Document**: Write a design proposal
2. **Implement Interface**: Follow `AttentionMechanism` trait
3. **Add Tests**: Comprehensive test coverage
4. **Benchmark**: Compare with existing methods
5. **Document**: API docs and usage examples

### Hardware Support

Adding support for new hardware:

1. **Abstract Interface**: Use hardware abstraction layer
2. **Feature Flag**: Add cargo feature
3. **Conditional Compilation**: Use `#[cfg]` attributes
4. **Testing**: Test on target hardware
5. **CI Integration**: Add to CI matrix

## Release Process

### Version Numbering

We follow Semantic Versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes

### Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in Cargo.toml
- [ ] Release notes drafted
- [ ] Performance benchmarks run
- [ ] Security audit completed

## Legal

### License

By contributing, you agree that your contributions will be licensed under the same license as the project (Apache 2.0).

### Developer Certificate of Origin

By contributing, you certify that:

1. The contribution is your original work
2. You have the right to submit it
3. You understand it will be public
4. You grant the project license rights

## Thank You!

Thank you for contributing to Long-Context Attention! Your efforts help make efficient attention mechanisms accessible to the community.