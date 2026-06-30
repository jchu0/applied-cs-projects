//! Token types for the Python lexer.

use crate::Span;

/// Token types.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    // Literals
    Integer(i64),
    Float(f64),
    String(String),
    True,
    False,
    None,

    // Identifier
    Identifier(String),

    // Keywords
    Def,
    Class,
    If,
    Elif,
    Else,
    While,
    For,
    In,
    Return,
    Try,
    Except,
    Finally,
    Raise,
    Import,
    From,
    As,
    And,
    Or,
    Not,
    Is,
    Lambda,
    Global,
    Nonlocal,
    Pass,
    Break,
    Continue,
    Yield,
    Async,
    Await,
    With,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    DoubleSlash,
    Percent,
    DoubleStar,
    At,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Assignment
    Assign,
    PlusAssign,
    MinusAssign,
    StarAssign,
    SlashAssign,
    PercentAssign,
    DoubleStarAssign,
    DoubleSlashAssign,

    // Delimiters
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Semicolon,
    Dot,
    Arrow,

    // Indentation
    Indent,
    Dedent,
    Newline,

    // Special
    Eof,
}

/// A token with its type and source location.
#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub lexeme: String,
    pub span: Span,
}

impl Token {
    pub fn new(token_type: TokenType, lexeme: String, span: Span) -> Self {
        Self {
            token_type,
            lexeme,
            span,
        }
    }

    pub fn line(&self) -> usize {
        self.span.line
    }
}

impl TokenType {
    /// Get keyword from string.
    pub fn keyword(s: &str) -> Option<TokenType> {
        match s {
            "def" => Some(TokenType::Def),
            "class" => Some(TokenType::Class),
            "if" => Some(TokenType::If),
            "elif" => Some(TokenType::Elif),
            "else" => Some(TokenType::Else),
            "while" => Some(TokenType::While),
            "for" => Some(TokenType::For),
            "in" => Some(TokenType::In),
            "return" => Some(TokenType::Return),
            "try" => Some(TokenType::Try),
            "except" => Some(TokenType::Except),
            "finally" => Some(TokenType::Finally),
            "raise" => Some(TokenType::Raise),
            "import" => Some(TokenType::Import),
            "from" => Some(TokenType::From),
            "as" => Some(TokenType::As),
            "and" => Some(TokenType::And),
            "or" => Some(TokenType::Or),
            "not" => Some(TokenType::Not),
            "is" => Some(TokenType::Is),
            "lambda" => Some(TokenType::Lambda),
            "global" => Some(TokenType::Global),
            "nonlocal" => Some(TokenType::Nonlocal),
            "pass" => Some(TokenType::Pass),
            "break" => Some(TokenType::Break),
            "continue" => Some(TokenType::Continue),
            "True" => Some(TokenType::True),
            "False" => Some(TokenType::False),
            "None" => Some(TokenType::None),
            "yield" => Some(TokenType::Yield),
            "async" => Some(TokenType::Async),
            "await" => Some(TokenType::Await),
            "with" => Some(TokenType::With),
            _ => None,
        }
    }
}
