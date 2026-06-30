//! Abstract Syntax Tree nodes.

use crate::Span;

/// Expression node.
#[derive(Debug, Clone)]
pub enum Expr {
    // Literals
    Integer(i64),
    Float(f64),
    String(String),
    Bool(bool),
    None,

    // Collections
    List(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),
    Tuple(Vec<Expr>),

    // Operations
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    UnaryOp {
        op: UnaryOp,
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
        target: String,
        iter: Box<Expr>,
        condition: Option<Box<Expr>>,
    },

    // Lambda
    Lambda {
        params: Vec<Param>,
        body: Box<Expr>,
    },

    // Conditional expression
    IfExpr {
        test: Box<Expr>,
        body: Box<Expr>,
        orelse: Box<Expr>,
    },

    // Generator expressions
    Yield(Option<Box<Expr>>),
    YieldFrom(Box<Expr>),

    // Async expressions
    Await(Box<Expr>),
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    And,
    Or,
}

impl BinaryOp {
    /// Get precedence for Pratt parser.
    pub fn precedence(&self) -> u8 {
        match self {
            BinaryOp::Or => 1,
            BinaryOp::And => 2,
            BinaryOp::Add | BinaryOp::Sub => 4,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::FloorDiv | BinaryOp::Mod => 5,
            BinaryOp::Pow => 6,
        }
    }

    /// Check if right-associative.
    pub fn is_right_associative(&self) -> bool {
        matches!(self, BinaryOp::Pow)
    }
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// Comparison operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Is,
    IsNot,
    In,
    NotIn,
}

/// Function parameter.
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub default: Option<Expr>,
}

/// Statement node.
#[derive(Debug, Clone)]
pub enum Stmt {
    // Simple statements
    Expr(Expr),
    Assign {
        targets: Vec<Expr>,
        value: Expr,
    },
    AugAssign {
        target: Expr,
        op: BinaryOp,
        value: Expr,
    },
    Return(Option<Expr>),
    Raise(Option<Expr>),
    Pass,
    Break,
    Continue,
    Global(Vec<String>),
    Nonlocal(Vec<String>),

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
    },
    For {
        target: String,
        iter: Expr,
        body: Vec<Stmt>,
    },
    FunctionDef {
        name: String,
        params: Vec<Param>,
        body: Vec<Stmt>,
        decorators: Vec<Expr>,
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
    },

    // Generator/async statements
    AsyncFunctionDef {
        name: String,
        params: Vec<Param>,
        body: Vec<Stmt>,
        decorators: Vec<Expr>,
    },
    AsyncFor {
        target: String,
        iter: Expr,
        body: Vec<Stmt>,
    },
    AsyncWith {
        items: Vec<WithItem>,
        body: Vec<Stmt>,
    },
    With {
        items: Vec<WithItem>,
        body: Vec<Stmt>,
    },
}

/// With statement context item.
#[derive(Debug, Clone)]
pub struct WithItem {
    pub context_expr: Expr,
    pub optional_vars: Option<String>,
}

/// Exception handler.
#[derive(Debug, Clone)]
pub struct ExceptHandler {
    pub exception_type: Option<String>,
    pub name: Option<String>,
    pub body: Vec<Stmt>,
}

/// Import alias.
#[derive(Debug, Clone)]
pub struct Alias {
    pub name: String,
    pub asname: Option<String>,
}

/// A complete program (module).
#[derive(Debug, Clone)]
pub struct Module {
    pub body: Vec<Stmt>,
}
