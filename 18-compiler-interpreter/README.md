# Python Subset Compiler & Interpreter

A compiler and interpreter for a Python subset, implemented in Rust. Supports core Python features including classes, functions, control flow, and basic data structures.

## Features

- **Core Language Support**: Variables, functions, classes, control flow
- **Data Types**: int, float, str, list, dict, tuple, set
- **Object-Oriented**: Classes, inheritance, methods
- **Functional Features**: Lambda, closures, decorators
- **Exception Handling**: try/except/finally blocks
- **Built-in Functions**: print, len, range, type, etc.
- **Bytecode Compilation**: Compiles to custom bytecode format
- **REPL**: Interactive interpreter mode

## Quick Start

```rust
use compiler_interpreter::Interpreter;

fn main() {
    let code = r#"
def greet(name):
    return f"Hello, {name}!"

message = greet("World")
print(message)
"#;

    let mut interpreter = Interpreter::new();
    interpreter.execute(code).unwrap();
}
```

## Installation

```toml
[dependencies]
compiler-interpreter = "0.1.0"
```

## Supported Python Features

### Basic Operations
```python
# Arithmetic
x = 10 + 20 * 3
y = x ** 2 / 5

# Strings
name = "Python"
greeting = f"Hello, {name}!"

# Lists
numbers = [1, 2, 3]
squares = [x**2 for x in numbers]
```

### Functions & Classes
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hi, I'm {self.name}"
```

### Control Flow
```python
# Conditionals
if x > 0:
    print("Positive")
elif x < 0:
    print("Negative")
else:
    print("Zero")

# Loops
for i in range(10):
    if i % 2 == 0:
        continue
    print(i)

while condition:
    # do something
    break
```

## Architecture

- **Lexer**: Tokenizes Python source code
- **Parser**: Builds Abstract Syntax Tree (AST)
- **Compiler**: Converts AST to bytecode
- **VM**: Executes bytecode instructions
- **Built-ins**: Standard library functions

## Testing

```bash
# Run all tests
cargo test

# Run specific test suite
cargo test lexer_tests
cargo test parser_tests
cargo test vm_tests
cargo test integration_tests

# Test coverage (60%+)
cargo tarpaulin
```

## Performance

Benchmarks (relative to CPython 3.9):

| Operation | Performance |
|-----------|-------------|
| Arithmetic | 0.8x |
| Function Calls | 0.6x |
| Object Creation | 0.5x |
| List Operations | 0.7x |

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Language Guide](docs/LANGUAGE.md)

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## Limitations

This is a subset implementation. Not supported:
- async/await
- metaclasses
- complex imports
- advanced decorators
- some built-in modules

## License

MIT/Apache-2.0