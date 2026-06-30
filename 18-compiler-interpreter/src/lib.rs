//! Python Subset Compiler/Interpreter.
//!
//! A complete compilation pipeline from source code to execution,
//! including lexical analysis, parsing, semantic analysis, bytecode
//! generation, and a virtual machine with garbage collection.

pub mod token;
pub mod lexer;
pub mod ast;
pub mod parser;
pub mod compiler;
pub mod vm;
pub mod value;
pub mod builtins;

use thiserror::Error;

/// Compiler/interpreter errors.
#[derive(Error, Debug)]
pub enum Error {
    #[error("Lexer error at line {line}, column {column}: {message}")]
    Lexer {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("Parse error at line {line}: {message}")]
    Parse { message: String, line: usize },

    #[error("Semantic error: {0}")]
    Semantic(String),

    #[error("Compile error: {0}")]
    Compile(String),

    #[error("Runtime error: {0}")]
    Runtime(String),

    #[error("Type error: {0}")]
    Type(String),

    #[error("Name error: {0}")]
    Name(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Key error: {0}")]
    Key(String),

    #[error("Value error: {0}")]
    Value(String),

    #[error("Attribute error: {0}")]
    Attribute(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for compiler operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Source location span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

impl Span {
    pub fn new(start: usize, end: usize, line: usize, column: usize) -> Self {
        Self {
            start,
            end,
            line,
            column,
        }
    }
}

/// Compile and run Python source code.
pub fn run(source: &str) -> Result<value::Value> {
    let tokens = lexer::Lexer::new(source).tokenize()?;
    let ast = parser::Parser::new(tokens).parse()?;
    let code = compiler::Compiler::new().compile(&ast)?;
    let mut vm = vm::VM::new();
    vm.run(&code)
}
