# Compiler/Interpreter for Python Subset

## Project Overview

A complete compiler/interpreter implementation for a substantial subset of Python. This project covers the entire compilation pipeline from source code to execution, including lexical analysis, parsing, semantic analysis, bytecode generation, and a virtual machine with garbage collection. The goal is to understand how programming languages are implemented from the ground up.

## Architecture

### Compilation Pipeline

```
┌─────────────┐   ┌─────────┐   ┌────────┐   ┌──────────┐   ┌──────────┐
│Source Code  │──►│  Lexer  │──►│ Parser │──►│ Analyzer │──►│ Compiler │
└─────────────┘   └─────────┘   └────────┘   └──────────┘   └──────────┘
                       │             │            │              │
                       ▼             ▼            ▼              ▼
                   [Tokens]       [AST]    [Typed AST]    [Bytecode]
                                                               │
                                                               ▼
                                                        ┌──────────┐
                                                        │    VM    │
                                                        │ ┌──────┐ │
                                                        │ │Stack │ │
                                                        │ │Heap  │ │
                                                        │ │GC    │ │
                                                        │ └──────┘ │
                                                        └──────────┘
```

### Core Components

#### 1. Lexer (Tokenizer)
- Breaks source into tokens
- Handles indentation-based scoping
- Manages string interpolation
- Reports lexical errors

#### 2. Parser
- Builds Abstract Syntax Tree
- Recursive descent or Pratt parsing
- Operator precedence handling
- Syntax error recovery

#### 3. Semantic Analyzer
- Name resolution
- Type inference/checking
- Scope management
- Control flow analysis

#### 4. Bytecode Compiler
- AST to bytecode transformation
- Constant pool management
- Local variable allocation
- Closure compilation

#### 5. Virtual Machine
- Bytecode interpreter
- Stack management
- Object system
- Garbage collection

## Language Specification

### Supported Syntax

```python
# Variables and basic types
x = 42
name = "Python"
pi = 3.14
flag = True
nothing = None

# Collections
numbers = [1, 2, 3]
point = (10, 20)
person = {"name": "Alice", "age": 30}
unique = {1, 2, 3}

# Control flow
if condition:
    do_something()
elif other:
    do_other()
else:
    default()

while count > 0:
    count -= 1

for item in collection:
    process(item)

for i in range(10):
    print(i)

# Functions
def add(a, b):
    return a + b

def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Closures
def make_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

# Classes
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

# Inheritance
class Point3D(Point):
    def __init__(self, x, y, z):
        super().__init__(x, y)
        self.z = z

# Exception handling
try:
    risky_operation()
except ValueError as e:
    handle_error(e)
finally:
    cleanup()

# List comprehensions
squares = [x**2 for x in range(10) if x % 2 == 0]

# Lambda expressions
double = lambda x: x * 2
```

### Type System

```rust
enum Type {
    None,
    Bool,
    Int,
    Float,
    String,
    List(Box<Type>),
    Dict(Box<Type>, Box<Type>),
    Set(Box<Type>),
    Tuple(Vec<Type>),
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
    },
    Class {
        name: String,
        fields: HashMap<String, Type>,
        methods: HashMap<String, Type>,
    },
    Any,  // For dynamic typing
}
```

## Lexer Implementation

### Token Types

```rust
enum TokenType {
    // Literals
    Integer(i64),
    Float(f64),
    String(String),
    FString(Vec<FStringPart>),
    True, False, None,

    // Identifiers and keywords
    Identifier(String),
    Def, Class, If, Elif, Else,
    While, For, In, Return,
    Try, Except, Finally, Raise,
    Import, From, As,
    And, Or, Not, Is,
    Lambda, Yield, Async, Await,
    Global, Nonlocal,
    Pass, Break, Continue,

    // Operators
    Plus, Minus, Star, Slash, DoubleSlash,
    Percent, DoubleStar, At,
    Eq, Ne, Lt, Le, Gt, Ge,
    Assign, PlusAssign, MinusAssign, /* ... */

    // Delimiters
    LParen, RParen, LBracket, RBracket, LBrace, RBrace,
    Comma, Colon, Semicolon, Dot, Arrow,

    // Indentation
    Indent,
    Dedent,
    Newline,

    // Special
    EOF,
}

struct Token {
    token_type: TokenType,
    lexeme: String,
    line: usize,
    column: usize,
}
```

### Indentation Handling

```rust
struct Lexer {
    source: Vec<char>,
    tokens: Vec<Token>,
    indent_stack: Vec<usize>,  // Stack of indentation levels
    at_line_start: bool,
    // ...
}

impl Lexer {
    fn handle_indentation(&mut self) {
        let spaces = self.count_leading_spaces();
        let current_indent = *self.indent_stack.last().unwrap();

        if spaces > current_indent {
            // New indentation level
            self.indent_stack.push(spaces);
            self.emit(TokenType::Indent);
        } else if spaces < current_indent {
            // One or more dedents
            while let Some(&level) = self.indent_stack.last() {
                if level > spaces {
                    self.indent_stack.pop();
                    self.emit(TokenType::Dedent);
                } else {
                    break;
                }
            }
            if self.indent_stack.last() != Some(&spaces) {
                self.error("Inconsistent indentation");
            }
        }
    }
}
```

## Parser Implementation

### AST Node Types

```rust
enum Expr {
    // Literals
    Integer(i64),
    Float(f64),
    String(String),
    Bool(bool),
    None,

    // Collections
    List(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),
    Set(Vec<Expr>),
    Tuple(Vec<Expr>),

    // Operations
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expr>,
    },
    Compare {
        left: Box<Expr>,
        ops: Vec<(CompareOp, Expr)>,
    },

    // Access
    Identifier(String),
    Attribute {
        value: Box<Expr>,
        attr: String,
    },
    Subscript {
        value: Box<Expr>,
        index: Box<Expr>,
    },

    // Calls
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
        kwargs: Vec<(String, Expr)>,
    },

    // Comprehensions
    ListComp {
        element: Box<Expr>,
        generators: Vec<Comprehension>,
    },

    // Lambda
    Lambda {
        params: Vec<Param>,
        body: Box<Expr>,
    },

    // Conditional
    IfExpr {
        test: Box<Expr>,
        body: Box<Expr>,
        orelse: Box<Expr>,
    },
}

enum Stmt {
    // Simple statements
    Expr(Expr),
    Assign {
        targets: Vec<Expr>,
        value: Expr,
    },
    AugAssign {
        target: Expr,
        op: BinaryOperator,
        value: Expr,
    },
    Return(Option<Expr>),
    Raise(Option<Expr>),
    Pass,
    Break,
    Continue,

    // Compound statements
    If {
        test: Expr,
        body: Vec<Stmt>,
        elif_clauses: Vec<(Expr, Vec<Stmt>)>,
        else_body: Vec<Stmt>,
    },
    While {
        test: Expr,
        body: Vec<Stmt>,
        else_body: Vec<Stmt>,
    },
    For {
        target: Expr,
        iter: Expr,
        body: Vec<Stmt>,
        else_body: Vec<Stmt>,
    },
    FunctionDef {
        name: String,
        params: Vec<Param>,
        body: Vec<Stmt>,
        decorators: Vec<Expr>,
        returns: Option<Type>,
    },
    ClassDef {
        name: String,
        bases: Vec<Expr>,
        body: Vec<Stmt>,
        decorators: Vec<Expr>,
    },
    Try {
        body: Vec<Stmt>,
        handlers: Vec<ExceptHandler>,
        else_body: Vec<Stmt>,
        finally_body: Vec<Stmt>,
    },
    Import(Vec<Alias>),
    ImportFrom {
        module: String,
        names: Vec<Alias>,
        level: usize,
    },
}
```

### Pratt Parser for Expressions

```rust
impl Parser {
    fn parse_expression(&mut self, min_precedence: u8) -> Result<Expr> {
        let mut left = self.parse_prefix()?;

        while let Some(op) = self.peek_binary_op() {
            let precedence = op.precedence();
            if precedence < min_precedence {
                break;
            }

            self.advance();
            let right = self.parse_expression(
                if op.is_right_associative() {
                    precedence
                } else {
                    precedence + 1
                }
            )?;

            left = Expr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_prefix(&mut self) -> Result<Expr> {
        match self.current_token() {
            Token::Integer(n) => Ok(Expr::Integer(n)),
            Token::Identifier(name) => Ok(Expr::Identifier(name)),
            Token::LParen => self.parse_grouped_or_tuple(),
            Token::LBracket => self.parse_list(),
            Token::LBrace => self.parse_dict_or_set(),
            Token::Minus | Token::Not => self.parse_unary(),
            Token::Lambda => self.parse_lambda(),
            _ => Err(ParseError::UnexpectedToken),
        }
    }
}
```

## Semantic Analysis

### Symbol Table

```rust
struct SymbolTable {
    scopes: Vec<Scope>,
}

struct Scope {
    symbols: HashMap<String, Symbol>,
    parent: Option<usize>,
    scope_type: ScopeType,
}

enum ScopeType {
    Global,
    Function,
    Class,
    Comprehension,
}

struct Symbol {
    name: String,
    symbol_type: Type,
    is_initialized: bool,
    is_captured: bool,  // For closures
    declaration_site: Span,
}

impl SymbolTable {
    fn define(&mut self, name: &str, symbol_type: Type) -> Result<()>;
    fn lookup(&self, name: &str) -> Option<&Symbol>;
    fn resolve(&self, name: &str) -> Option<(usize, &Symbol)>;  // Returns scope depth
    fn enter_scope(&mut self, scope_type: ScopeType);
    fn exit_scope(&mut self) -> Scope;
}
```

### Type Checker

```rust
struct TypeChecker {
    symbol_table: SymbolTable,
    current_function_return: Option<Type>,
    in_loop: bool,
}

impl TypeChecker {
    fn check_expr(&mut self, expr: &Expr) -> Result<Type> {
        match expr {
            Expr::BinaryOp { left, op, right } => {
                let left_type = self.check_expr(left)?;
                let right_type = self.check_expr(right)?;
                self.check_binary_op(&left_type, op, &right_type)
            }
            Expr::Call { func, args, kwargs } => {
                let func_type = self.check_expr(func)?;
                self.check_call(&func_type, args, kwargs)
            }
            Expr::Attribute { value, attr } => {
                let value_type = self.check_expr(value)?;
                self.check_attribute(&value_type, attr)
            }
            // ...
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        match stmt {
            Stmt::FunctionDef { name, params, body, returns, .. } => {
                self.symbol_table.enter_scope(ScopeType::Function);
                for param in params {
                    self.symbol_table.define(&param.name, param.annotation.clone())?;
                }
                self.current_function_return = returns.clone();
                for stmt in body {
                    self.check_stmt(stmt)?;
                }
                self.symbol_table.exit_scope();
                Ok(())
            }
            Stmt::Return(expr) => {
                let return_type = self.current_function_return.clone()
                    .ok_or(SemanticError::ReturnOutsideFunction)?;
                if let Some(e) = expr {
                    let expr_type = self.check_expr(e)?;
                    self.check_assignable(&expr_type, &return_type)?;
                }
                Ok(())
            }
            // ...
        }
    }
}
```

## Bytecode Design

### Instruction Set

```rust
enum OpCode {
    // Stack operations
    LoadConst(u16),      // Push constant
    LoadName(u16),       // Load global
    LoadFast(u16),       // Load local
    LoadDeref(u16),      // Load closure variable
    StoreName(u16),      // Store global
    StoreFast(u16),      // Store local
    StoreDeref(u16),     // Store closure variable
    LoadAttr(u16),       // Load attribute
    StoreAttr(u16),      // Store attribute

    // Arithmetic
    BinaryAdd,
    BinarySubtract,
    BinaryMultiply,
    BinaryDivide,
    BinaryFloorDivide,
    BinaryModulo,
    BinaryPower,
    UnaryNegate,
    UnaryNot,

    // Comparison
    CompareEqual,
    CompareNotEqual,
    CompareLess,
    CompareLessEqual,
    CompareGreater,
    CompareGreaterEqual,
    CompareIs,
    CompareIsNot,
    CompareIn,
    CompareNotIn,

    // Control flow
    Jump(i16),           // Relative jump
    JumpIfTrue(i16),
    JumpIfFalse(i16),
    JumpIfTrueOrPop(i16),
    JumpIfFalseOrPop(i16),
    PopJumpIfTrue(i16),
    PopJumpIfFalse(i16),

    // Functions
    Call(u8),            // Call with n positional args
    CallKw(u8),          // Call with keyword args
    Return,
    MakeFunction(u8),    // Create function with flags
    MakeClosure(u8),

    // Collections
    BuildList(u16),
    BuildDict(u16),
    BuildSet(u16),
    BuildTuple(u16),
    ListAppend,
    SetAdd,
    DictSetItem,
    BinarySubscript,
    StoreSubscript,

    // Iteration
    GetIter,
    ForIter(i16),        // Jump if iterator exhausted

    // Class
    LoadBuildClass,
    LoadMethod(u16),
    CallMethod(u8),

    // Exception
    SetupExcept(i16),
    SetupFinally(i16),
    PopExcept,
    Raise,

    // Misc
    Pop,
    Dup,
    Rot2,
    Rot3,
    Nop,
}

struct CodeObject {
    name: String,
    bytecode: Vec<u8>,
    constants: Vec<Value>,
    names: Vec<String>,
    varnames: Vec<String>,
    freevars: Vec<String>,
    cellvars: Vec<String>,
    arg_count: u16,
    kwonly_arg_count: u16,
    stack_size: u16,
    line_numbers: Vec<(usize, usize)>,  // (bytecode offset, line)
}
```

### Bytecode Compiler

```rust
struct Compiler {
    code: CodeObject,
    scopes: Vec<CompilerScope>,
}

struct CompilerScope {
    locals: HashMap<String, u16>,
    cells: HashMap<String, u16>,
    frees: HashMap<String, u16>,
}

impl Compiler {
    fn compile_expr(&mut self, expr: &Expr) -> Result<()> {
        match expr {
            Expr::Integer(n) => {
                let idx = self.add_constant(Value::Int(*n));
                self.emit(OpCode::LoadConst(idx));
            }
            Expr::BinaryOp { left, op, right } => {
                self.compile_expr(left)?;
                self.compile_expr(right)?;
                self.emit(op.to_opcode());
            }
            Expr::Call { func, args, kwargs } => {
                self.compile_expr(func)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                if kwargs.is_empty() {
                    self.emit(OpCode::Call(args.len() as u8));
                } else {
                    // Handle keyword args
                }
            }
            // ...
        }
        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        match stmt {
            Stmt::If { test, body, elif_clauses, else_body } => {
                self.compile_expr(test)?;
                let jump_to_else = self.emit_jump(OpCode::PopJumpIfFalse(0));

                for stmt in body {
                    self.compile_stmt(stmt)?;
                }
                let jump_to_end = self.emit_jump(OpCode::Jump(0));

                self.patch_jump(jump_to_else);
                // ... compile elif and else

                self.patch_jump(jump_to_end);
                Ok(())
            }
            Stmt::While { test, body, .. } => {
                let loop_start = self.current_offset();
                self.compile_expr(test)?;
                let exit_jump = self.emit_jump(OpCode::PopJumpIfFalse(0));

                for stmt in body {
                    self.compile_stmt(stmt)?;
                }
                self.emit_loop(loop_start);
                self.patch_jump(exit_jump);
                Ok(())
            }
            // ...
        }
    }
}
```

## Virtual Machine

### Stack-Based VM

```rust
struct VM {
    stack: Vec<Value>,
    frames: Vec<CallFrame>,
    heap: Heap,
    globals: HashMap<String, Value>,
}

struct CallFrame {
    code: Rc<CodeObject>,
    ip: usize,              // Instruction pointer
    bp: usize,              // Base pointer (stack frame start)
    closure: Option<Rc<Closure>>,
}

impl VM {
    fn run(&mut self) -> Result<Value> {
        loop {
            let instruction = self.read_instruction();

            match instruction {
                OpCode::LoadConst(idx) => {
                    let value = self.current_frame().code.constants[idx as usize].clone();
                    self.push(value);
                }
                OpCode::BinaryAdd => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(self.add(&a, &b)?);
                }
                OpCode::Call(arg_count) => {
                    let callee = self.peek(arg_count as usize);
                    match callee {
                        Value::Function(func) => {
                            let frame = CallFrame {
                                code: func.code.clone(),
                                ip: 0,
                                bp: self.stack.len() - arg_count as usize,
                                closure: func.closure.clone(),
                            };
                            self.frames.push(frame);
                        }
                        Value::NativeFunction(native) => {
                            let args = self.pop_n(arg_count as usize);
                            let result = (native.func)(&args)?;
                            self.push(result);
                        }
                        _ => return Err(RuntimeError::NotCallable),
                    }
                }
                OpCode::Return => {
                    let result = self.pop();
                    let frame = self.frames.pop().unwrap();
                    self.stack.truncate(frame.bp);
                    if self.frames.is_empty() {
                        return Ok(result);
                    }
                    self.push(result);
                }
                // ... other opcodes
            }
        }
    }
}
```

### Object System

```rust
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(Rc<String>),
    List(Rc<RefCell<Vec<Value>>>),
    Dict(Rc<RefCell<HashMap<HashValue, Value>>>),
    Set(Rc<RefCell<HashSet<HashValue>>>),
    Tuple(Rc<Vec<Value>>),
    Function(Rc<Function>),
    NativeFunction(Rc<NativeFunction>),
    Class(Rc<Class>),
    Instance(Rc<RefCell<Instance>>),
    BoundMethod(Rc<BoundMethod>),
    Iterator(Box<dyn Iterator>),
}

struct Function {
    code: Rc<CodeObject>,
    closure: Option<Rc<Closure>>,
    defaults: Vec<Value>,
}

struct Class {
    name: String,
    bases: Vec<Rc<Class>>,
    methods: HashMap<String, Value>,
}

struct Instance {
    class: Rc<Class>,
    fields: HashMap<String, Value>,
}
```

### Garbage Collection

```rust
// Mark-and-sweep garbage collector
struct GC {
    heap: Vec<Box<dyn GcObject>>,
    gray_stack: Vec<*mut dyn GcObject>,
    bytes_allocated: usize,
    next_gc: usize,
}

trait GcObject {
    fn mark(&mut self);
    fn trace(&self, gray_stack: &mut Vec<*mut dyn GcObject>);
    fn size(&self) -> usize;
}

impl GC {
    fn collect(&mut self, roots: &[Value]) {
        // Mark phase
        for root in roots {
            self.mark_value(root);
        }

        // Trace phase
        while let Some(obj) = self.gray_stack.pop() {
            unsafe { (*obj).trace(&mut self.gray_stack) };
        }

        // Sweep phase
        self.heap.retain(|obj| {
            if obj.is_marked() {
                obj.unmark();
                true
            } else {
                self.bytes_allocated -= obj.size();
                false
            }
        });

        self.next_gc = self.bytes_allocated * 2;
    }
}
```

## Enterprise Features

### Error Reporting with Hints

```rust
struct Diagnostic {
    level: DiagnosticLevel,
    message: String,
    span: Span,
    hints: Vec<String>,
    notes: Vec<(Span, String)>,
}

fn format_error(source: &str, diag: &Diagnostic) -> String {
    // Example output:
    // error[E0001]: undefined variable 'foo'
    //   --> main.py:10:5
    //    |
    // 10 |     print(foo)
    //    |           ^^^ not found in this scope
    //    |
    //    = help: did you mean 'food'?
    //    = note: variables must be defined before use
}
```

### Standard Library

```rust
// Built-in functions
fn builtin_print(args: &[Value]) -> Result<Value>;
fn builtin_len(args: &[Value]) -> Result<Value>;
fn builtin_range(args: &[Value]) -> Result<Value>;
fn builtin_input(args: &[Value]) -> Result<Value>;
fn builtin_type(args: &[Value]) -> Result<Value>;
fn builtin_isinstance(args: &[Value]) -> Result<Value>;
fn builtin_int(args: &[Value]) -> Result<Value>;
fn builtin_float(args: &[Value]) -> Result<Value>;
fn builtin_str(args: &[Value]) -> Result<Value>;
fn builtin_list(args: &[Value]) -> Result<Value>;
fn builtin_dict(args: &[Value]) -> Result<Value>;
fn builtin_set(args: &[Value]) -> Result<Value>;
fn builtin_tuple(args: &[Value]) -> Result<Value>;
fn builtin_iter(args: &[Value]) -> Result<Value>;
fn builtin_next(args: &[Value]) -> Result<Value>;
fn builtin_enumerate(args: &[Value]) -> Result<Value>;
fn builtin_zip(args: &[Value]) -> Result<Value>;
fn builtin_map(args: &[Value]) -> Result<Value>;
fn builtin_filter(args: &[Value]) -> Result<Value>;
fn builtin_sorted(args: &[Value]) -> Result<Value>;
fn builtin_reversed(args: &[Value]) -> Result<Value>;
fn builtin_sum(args: &[Value]) -> Result<Value>;
fn builtin_min(args: &[Value]) -> Result<Value>;
fn builtin_max(args: &[Value]) -> Result<Value>;
fn builtin_abs(args: &[Value]) -> Result<Value>;
fn builtin_round(args: &[Value]) -> Result<Value>;
fn builtin_open(args: &[Value]) -> Result<Value>;
```

### REPL

```rust
struct Repl {
    vm: VM,
    history: Vec<String>,
}

impl Repl {
    fn run(&mut self) {
        loop {
            let input = self.read_multiline();
            if input.is_empty() {
                continue;
            }

            match self.eval(&input) {
                Ok(value) => {
                    if value != Value::None {
                        println!("{}", self.repr(&value));
                    }
                }
                Err(e) => {
                    eprintln!("{}", e);
                }
            }

            self.history.push(input);
        }
    }

    fn read_multiline(&mut self) -> String {
        // Handle multiline input for functions, classes, etc.
        // Continue reading if line ends with ':'
    }
}
```

## Performance Considerations

### Bytecode Optimizations

```rust
// Peephole optimization
fn optimize_bytecode(code: &mut Vec<OpCode>) {
    // Constant folding
    // LoadConst 2, LoadConst 3, BinaryAdd -> LoadConst 5

    // Dead code elimination
    // Remove unreachable code after unconditional jumps

    // Jump optimization
    // Jump to Jump -> Jump to final target

    // Strength reduction
    // x * 2 -> x + x (if faster)
}
```

### VM Optimizations

```rust
// Computed goto (requires unsafe)
// Direct threading instead of switch dispatch

// Inline caching for attribute access
struct InlineCache {
    type_id: TypeId,
    offset: usize,
}

// NaN boxing for compact value representation
// Store small integers and special values in pointer space
```

### Benchmarks

| Benchmark | Target | CPython 3.x |
|-----------|--------|-------------|
| Fibonacci(35) | < 2s | ~3s |
| List operations | < 100ms | ~150ms |
| Dict operations | < 50ms | ~80ms |
| Function calls | < 20ns/call | ~30ns/call |

## Implementation Phases

### Phase 1: Lexer (Weeks 1-2)
- [ ] Basic tokenization
- [ ] Indentation handling
- [ ] String literals (including f-strings)
- [ ] Number literals
- [ ] Comment handling
- [ ] Error reporting

### Phase 2: Parser (Weeks 3-4)
- [ ] AST data structures
- [ ] Expression parsing (Pratt)
- [ ] Statement parsing
- [ ] Function definitions
- [ ] Class definitions
- [ ] Error recovery

### Phase 3: Semantic Analysis (Weeks 5-6)
- [ ] Symbol table
- [ ] Name resolution
- [ ] Type checking (basic)
- [ ] Scope validation
- [ ] Control flow checking

### Phase 4: Bytecode Compiler (Weeks 7-9)
- [ ] Bytecode format design
- [ ] Expression compilation
- [ ] Statement compilation
- [ ] Function compilation
- [ ] Class compilation
- [ ] Closure compilation

### Phase 5: Virtual Machine (Weeks 10-12)
- [ ] Stack-based interpreter
- [ ] Object system
- [ ] Built-in operations
- [ ] Function calls
- [ ] Iteration protocol

### Phase 6: Garbage Collection (Week 13)
- [ ] Mark-and-sweep GC
- [ ] Root finding
- [ ] Cycle detection
- [ ] GC tuning

### Phase 7: Standard Library (Weeks 14-15)
- [ ] Built-in functions
- [ ] Built-in types methods
- [ ] File I/O
- [ ] Basic modules

### Phase 8: Enterprise Features (Weeks 16-17)
- [ ] REPL
- [ ] Error messages with hints
- [ ] Debug info
- [ ] Source maps

### Phase 9: Optimization (Weeks 18-20)
- [ ] Constant folding
- [ ] Dead code elimination
- [ ] Peephole optimization
- [ ] VM dispatch optimization

## Testing Strategy

### Lexer Tests

```rust
#[test]
fn test_indentation() {
    let tokens = lex("def foo():\n    pass");
    assert_tokens!(tokens, [
        Def, Identifier("foo"), LParen, RParen, Colon, Newline,
        Indent, Pass, Newline, Dedent
    ]);
}

#[test]
fn test_string_escapes() {
    let tokens = lex(r#""hello\nworld""#);
    assert_eq!(tokens[0], Token::String("hello\nworld".into()));
}
```

### Parser Tests

```rust
#[test]
fn test_parse_function() {
    let ast = parse("def add(a, b):\n    return a + b");
    match ast {
        Stmt::FunctionDef { name, params, body, .. } => {
            assert_eq!(name, "add");
            assert_eq!(params.len(), 2);
        }
        _ => panic!("Expected function definition"),
    }
}
```

### Integration Tests

```rust
#[test]
fn test_fibonacci() {
    let result = run_code(r#"
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
fib(10)
"#);
    assert_eq!(result, Value::Int(55));
}
```

### Conformance Tests

```rust
// Run Python test suite subset
// Compare output with CPython
```

## Stretch Goals

### JIT Compilation
- Use LLVM or Cranelift
- Hot function detection
- Type specialization
- Inline caching

### Advanced Optimizations
- SSA form IR
- Register allocation
- Loop optimizations
- Escape analysis

### Additional Features
- Generator functions
- Async/await
- Decorators
- Metaclasses
- Descriptors

## Technology Stack

- **Language**: Rust
- **Parser**: Hand-written or logos/pest
- **JIT**: LLVM/Cranelift (optional)
- **Testing**: Built-in test framework

## References

- [Crafting Interpreters](https://craftinginterpreters.com/)
- [CPython Internals](https://realpython.com/cpython-source-code-guide/)
- [Python Language Reference](https://docs.python.org/3/reference/)
- [Write Yourself a Scheme in 48 Hours](https://en.wikibooks.org/wiki/Write_Yourself_a_Scheme_in_48_Hours)
- [Engineering a Compiler](https://www.elsevier.com/books/engineering-a-compiler/cooper/978-0-12-815412-0)
