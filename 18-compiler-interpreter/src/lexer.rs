//! Lexical analyzer for Python source code.

use crate::token::{Token, TokenType};
use crate::{Error, Result, Span};

/// Lexer for tokenizing Python source code.
pub struct Lexer {
    source: Vec<char>,
    tokens: Vec<Token>,
    start: usize,
    current: usize,
    line: usize,
    column: usize,
    indent_stack: Vec<usize>,
    at_line_start: bool,
    paren_depth: usize,
}

impl Lexer {
    /// Create a new lexer.
    pub fn new(source: &str) -> Self {
        Self {
            source: source.chars().collect(),
            tokens: Vec::new(),
            start: 0,
            current: 0,
            line: 1,
            column: 1,
            indent_stack: vec![0],
            at_line_start: true,
            paren_depth: 0,
        }
    }

    /// Tokenize the source code.
    pub fn tokenize(mut self) -> Result<Vec<Token>> {
        while !self.is_at_end() {
            self.start = self.current;
            self.scan_token()?;
        }

        // Emit remaining dedents
        while self.indent_stack.len() > 1 {
            self.indent_stack.pop();
            self.emit(TokenType::Dedent);
        }

        self.emit(TokenType::Eof);
        Ok(self.tokens)
    }

    fn scan_token(&mut self) -> Result<()> {
        // Handle indentation at line start
        if self.at_line_start && self.paren_depth == 0 {
            self.handle_indentation()?;
        }

        // Return early if we've reached end of input
        if self.is_at_end() {
            return Ok(());
        }

        let c = self.advance();

        match c {
            // Skip whitespace (except newlines)
            ' ' | '\t' | '\r' => {}

            // Newlines
            '\n' => {
                if self.paren_depth == 0 {
                    self.emit(TokenType::Newline);
                }
                self.line += 1;
                self.column = 1;
                self.at_line_start = true;
                return Ok(()); // Return early to preserve at_line_start
            }

            // Comments
            '#' => {
                while !self.is_at_end() && self.peek() != '\n' {
                    self.advance();
                }
            }

            // Parentheses
            '(' => {
                self.paren_depth += 1;
                self.emit(TokenType::LParen);
            }
            ')' => {
                self.paren_depth = self.paren_depth.saturating_sub(1);
                self.emit(TokenType::RParen);
            }
            '[' => {
                self.paren_depth += 1;
                self.emit(TokenType::LBracket);
            }
            ']' => {
                self.paren_depth = self.paren_depth.saturating_sub(1);
                self.emit(TokenType::RBracket);
            }
            '{' => {
                self.paren_depth += 1;
                self.emit(TokenType::LBrace);
            }
            '}' => {
                self.paren_depth = self.paren_depth.saturating_sub(1);
                self.emit(TokenType::RBrace);
            }

            // Delimiters
            ',' => self.emit(TokenType::Comma),
            ':' => self.emit(TokenType::Colon),
            ';' => self.emit(TokenType::Semicolon),
            '.' => self.emit(TokenType::Dot),

            // Operators
            '+' => {
                if self.match_char('=') {
                    self.emit(TokenType::PlusAssign);
                } else {
                    self.emit(TokenType::Plus);
                }
            }
            '-' => {
                if self.match_char('=') {
                    self.emit(TokenType::MinusAssign);
                } else if self.match_char('>') {
                    self.emit(TokenType::Arrow);
                } else {
                    self.emit(TokenType::Minus);
                }
            }
            '*' => {
                if self.match_char('*') {
                    if self.match_char('=') {
                        self.emit(TokenType::DoubleStarAssign);
                    } else {
                        self.emit(TokenType::DoubleStar);
                    }
                } else if self.match_char('=') {
                    self.emit(TokenType::StarAssign);
                } else {
                    self.emit(TokenType::Star);
                }
            }
            '/' => {
                if self.match_char('/') {
                    if self.match_char('=') {
                        self.emit(TokenType::DoubleSlashAssign);
                    } else {
                        self.emit(TokenType::DoubleSlash);
                    }
                } else if self.match_char('=') {
                    self.emit(TokenType::SlashAssign);
                } else {
                    self.emit(TokenType::Slash);
                }
            }
            '%' => {
                if self.match_char('=') {
                    self.emit(TokenType::PercentAssign);
                } else {
                    self.emit(TokenType::Percent);
                }
            }
            '@' => self.emit(TokenType::At),

            // Comparison and assignment
            '=' => {
                if self.match_char('=') {
                    self.emit(TokenType::Eq);
                } else {
                    self.emit(TokenType::Assign);
                }
            }
            '!' => {
                if self.match_char('=') {
                    self.emit(TokenType::Ne);
                } else {
                    return Err(self.error("Unexpected character '!'"));
                }
            }
            '<' => {
                if self.match_char('=') {
                    self.emit(TokenType::Le);
                } else {
                    self.emit(TokenType::Lt);
                }
            }
            '>' => {
                if self.match_char('=') {
                    self.emit(TokenType::Ge);
                } else {
                    self.emit(TokenType::Gt);
                }
            }

            // Strings
            '"' | '\'' => self.string(c)?,

            // Numbers
            c if c.is_ascii_digit() => self.number()?,

            // Identifiers and keywords
            c if c.is_alphabetic() || c == '_' => self.identifier(),

            _ => return Err(self.error(&format!("Unexpected character '{}'", c))),
        }

        self.at_line_start = false;
        Ok(())
    }

    fn handle_indentation(&mut self) -> Result<()> {
        let mut spaces = 0;

        while !self.is_at_end() && (self.peek() == ' ' || self.peek() == '\t') {
            if self.peek() == ' ' {
                spaces += 1;
            } else {
                spaces += 4; // Tab = 4 spaces
            }
            self.advance();
        }

        // Skip blank lines and comment-only lines
        if self.is_at_end() || self.peek() == '\n' || self.peek() == '#' {
            return Ok(());
        }

        let current_indent = *self.indent_stack.last().unwrap();

        if spaces > current_indent {
            self.indent_stack.push(spaces);
            self.emit(TokenType::Indent);
        } else if spaces < current_indent {
            while let Some(&level) = self.indent_stack.last() {
                if level > spaces {
                    self.indent_stack.pop();
                    self.emit(TokenType::Dedent);
                } else {
                    break;
                }
            }

            if self.indent_stack.last() != Some(&spaces) {
                return Err(self.error("Inconsistent indentation"));
            }
        }

        self.start = self.current;
        Ok(())
    }

    fn string(&mut self, quote: char) -> Result<()> {
        let mut value = String::new();

        while !self.is_at_end() && self.peek() != quote {
            if self.peek() == '\n' {
                self.line += 1;
                self.column = 1;
            }

            if self.peek() == '\\' {
                self.advance();
                let escaped = match self.advance() {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    '\'' => '\'',
                    '"' => '"',
                    '0' => '\0',
                    c => c,
                };
                value.push(escaped);
            } else {
                value.push(self.advance());
            }
        }

        if self.is_at_end() {
            return Err(self.error("Unterminated string"));
        }

        self.advance(); // Closing quote
        self.emit(TokenType::String(value));
        Ok(())
    }

    fn number(&mut self) -> Result<()> {
        while !self.is_at_end() && self.peek().is_ascii_digit() {
            self.advance();
        }

        let mut is_float = false;

        // Check for decimal part
        if !self.is_at_end() && self.peek() == '.' {
            if self.peek_next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                is_float = true;
                self.advance(); // Consume '.'
                while !self.is_at_end() && self.peek().is_ascii_digit() {
                    self.advance();
                }
            }
        }

        // Check for scientific notation (works for both integers and decimals)
        if !self.is_at_end() && (self.peek() == 'e' || self.peek() == 'E') {
            is_float = true;
            self.advance();
            if !self.is_at_end() && (self.peek() == '+' || self.peek() == '-') {
                self.advance();
            }
            while !self.is_at_end() && self.peek().is_ascii_digit() {
                self.advance();
            }
        }

        let lexeme: String = self.source[self.start..self.current].iter().collect();

        if is_float {
            let value: f64 = lexeme
                .parse()
                .map_err(|_| self.error("Invalid float literal"))?;
            self.emit(TokenType::Float(value));
            return Ok(());
        }

        let lexeme: String = self.source[self.start..self.current].iter().collect();
        let value: i64 = lexeme
            .parse()
            .map_err(|_| self.error("Invalid integer literal"))?;
        self.emit(TokenType::Integer(value));
        Ok(())
    }

    fn identifier(&mut self) {
        while !self.is_at_end() && (self.peek().is_alphanumeric() || self.peek() == '_') {
            self.advance();
        }

        let lexeme: String = self.source[self.start..self.current].iter().collect();

        let token_type = TokenType::keyword(&lexeme)
            .unwrap_or_else(|| TokenType::Identifier(lexeme));

        self.emit(token_type);
    }

    fn advance(&mut self) -> char {
        let c = self.source[self.current];
        self.current += 1;
        self.column += 1;
        c
    }

    fn peek(&self) -> char {
        if self.is_at_end() {
            '\0'
        } else {
            self.source[self.current]
        }
    }

    fn peek_next(&self) -> Option<char> {
        if self.current + 1 >= self.source.len() {
            None
        } else {
            Some(self.source[self.current + 1])
        }
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.is_at_end() || self.source[self.current] != expected {
            false
        } else {
            self.current += 1;
            self.column += 1;
            true
        }
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.source.len()
    }

    fn emit(&mut self, token_type: TokenType) {
        let lexeme: String = self.source[self.start..self.current].iter().collect();
        let span = Span::new(
            self.start,
            self.current,
            self.line,
            self.column - lexeme.len(),
        );
        self.tokens.push(Token::new(token_type, lexeme, span));
    }

    fn error(&self, message: &str) -> Error {
        Error::Lexer {
            message: message.to_string(),
            line: self.line,
            column: self.column,
        }
    }
}
