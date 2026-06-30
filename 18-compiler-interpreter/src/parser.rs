//! Parser for Python source code.

use crate::ast::*;
use crate::token::{Token, TokenType};
use crate::{Error, Result};

/// Parser for building AST from tokens.
pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    /// Create a new parser.
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, current: 0 }
    }

    /// Parse tokens into AST.
    pub fn parse(mut self) -> Result<Module> {
        let mut body = Vec::new();

        while !self.is_at_end() {
            self.skip_newlines();
            if self.is_at_end() {
                break;
            }
            body.push(self.statement()?);
        }

        Ok(Module { body })
    }

    fn statement(&mut self) -> Result<Stmt> {
        match self.peek().token_type {
            TokenType::At => self.decorated_def(),
            TokenType::Def => self.function_def(Vec::new()),
            TokenType::Async => self.async_stmt(Vec::new()),
            TokenType::Class => self.class_def(Vec::new()),
            TokenType::If => self.if_stmt(),
            TokenType::While => self.while_stmt(),
            TokenType::For => self.for_stmt(),
            TokenType::Try => self.try_stmt(),
            TokenType::With => self.with_stmt(),
            TokenType::Return => self.return_stmt(),
            TokenType::Raise => self.raise_stmt(),
            TokenType::Pass => {
                self.advance();
                self.expect_newline()?;
                Ok(Stmt::Pass)
            }
            TokenType::Break => {
                self.advance();
                self.expect_newline()?;
                Ok(Stmt::Break)
            }
            TokenType::Continue => {
                self.advance();
                self.expect_newline()?;
                Ok(Stmt::Continue)
            }
            TokenType::Global => self.global_stmt(),
            TokenType::Nonlocal => self.nonlocal_stmt(),
            TokenType::Import => self.import_stmt(),
            TokenType::From => self.from_import_stmt(),
            _ => self.expr_or_assign_stmt(),
        }
    }

    /// Parse @decorator annotations before function/class definitions.
    fn decorated_def(&mut self) -> Result<Stmt> {
        let mut decorators = Vec::new();

        // Collect all decorators
        while self.match_token(&TokenType::At) {
            let decorator = self.expression()?;
            decorators.push(decorator);
            self.expect_newline()?;
            self.skip_newlines();
        }

        // Parse the decorated definition
        match self.peek().token_type {
            TokenType::Def => self.function_def(decorators),
            TokenType::Async => self.async_stmt(decorators),
            TokenType::Class => self.class_def(decorators),
            _ => Err(self.error("Expected 'def', 'async def', or 'class' after decorator")),
        }
    }

    fn function_def(&mut self, decorators: Vec<Expr>) -> Result<Stmt> {
        self.expect(TokenType::Def)?;
        let name = self.expect_identifier()?;
        self.expect(TokenType::LParen)?;

        let mut params = Vec::new();
        if !self.check(&TokenType::RParen) {
            loop {
                let param_name = self.expect_identifier()?;
                let default = if self.match_token(&TokenType::Assign) {
                    Some(self.expression()?)
                } else {
                    None
                };
                params.push(Param {
                    name: param_name,
                    default,
                });

                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }

        self.expect(TokenType::RParen)?;
        self.expect(TokenType::Colon)?;
        let body = self.block()?;

        Ok(Stmt::FunctionDef { name, params, body, decorators })
    }

    fn async_stmt(&mut self, decorators: Vec<Expr>) -> Result<Stmt> {
        self.expect(TokenType::Async)?;
        match self.peek().token_type {
            TokenType::Def => {
                self.expect(TokenType::Def)?;
                let name = self.expect_identifier()?;
                self.expect(TokenType::LParen)?;

                let mut params = Vec::new();
                if !self.check(&TokenType::RParen) {
                    loop {
                        let param_name = self.expect_identifier()?;
                        let default = if self.match_token(&TokenType::Assign) {
                            Some(self.expression()?)
                        } else {
                            None
                        };
                        params.push(Param {
                            name: param_name,
                            default,
                        });

                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }

                self.expect(TokenType::RParen)?;
                self.expect(TokenType::Colon)?;
                let body = self.block()?;

                Ok(Stmt::AsyncFunctionDef { name, params, body, decorators })
            }
            TokenType::For => {
                self.expect(TokenType::For)?;
                let target = self.expect_identifier()?;
                self.expect(TokenType::In)?;
                let iter = self.expression()?;
                self.expect(TokenType::Colon)?;
                let body = self.block()?;

                Ok(Stmt::AsyncFor { target, iter, body })
            }
            TokenType::With => {
                self.expect(TokenType::With)?;
                let items = self.parse_with_items()?;
                self.expect(TokenType::Colon)?;
                let body = self.block()?;

                Ok(Stmt::AsyncWith { items, body })
            }
            _ => Err(self.error("Expected 'def', 'for', or 'with' after 'async'")),
        }
    }

    fn with_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenType::With)?;
        let items = self.parse_with_items()?;
        self.expect(TokenType::Colon)?;
        let body = self.block()?;

        Ok(Stmt::With { items, body })
    }

    fn parse_with_items(&mut self) -> Result<Vec<WithItem>> {
        let mut items = Vec::new();
        loop {
            let context_expr = self.expression()?;
            let optional_vars = if self.match_token(&TokenType::As) {
                Some(self.expect_identifier()?)
            } else {
                None
            };
            items.push(WithItem {
                context_expr,
                optional_vars,
            });
            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }
        Ok(items)
    }

    fn class_def(&mut self, decorators: Vec<Expr>) -> Result<Stmt> {
        self.expect(TokenType::Class)?;
        let name = self.expect_identifier()?;

        let mut bases = Vec::new();
        if self.match_token(&TokenType::LParen) {
            if !self.check(&TokenType::RParen) {
                loop {
                    bases.push(self.expression()?);
                    if !self.match_token(&TokenType::Comma) {
                        break;
                    }
                }
            }
            self.expect(TokenType::RParen)?;
        }

        self.expect(TokenType::Colon)?;
        let body = self.block()?;

        Ok(Stmt::ClassDef { name, bases, body, decorators })
    }

    fn if_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenType::If)?;
        let test = self.expression()?;
        self.expect(TokenType::Colon)?;
        let body = self.block()?;

        let mut elif_clauses = Vec::new();
        while self.match_token(&TokenType::Elif) {
            let elif_test = self.expression()?;
            self.expect(TokenType::Colon)?;
            let elif_body = self.block()?;
            elif_clauses.push((elif_test, elif_body));
        }

        let else_body = if self.match_token(&TokenType::Else) {
            self.expect(TokenType::Colon)?;
            self.block()?
        } else {
            Vec::new()
        };

        Ok(Stmt::If {
            test,
            body,
            elif_clauses,
            else_body,
        })
    }

    fn while_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenType::While)?;
        let test = self.expression()?;
        self.expect(TokenType::Colon)?;
        let body = self.block()?;

        Ok(Stmt::While { test, body })
    }

    fn for_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenType::For)?;
        let target = self.expect_identifier()?;
        self.expect(TokenType::In)?;
        let iter = self.expression()?;
        self.expect(TokenType::Colon)?;
        let body = self.block()?;

        Ok(Stmt::For { target, iter, body })
    }

    fn try_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenType::Try)?;
        self.expect(TokenType::Colon)?;
        let body = self.block()?;

        let mut handlers = Vec::new();
        while self.match_token(&TokenType::Except) {
            let exception_type = if !self.check(&TokenType::Colon) {
                Some(self.expect_identifier()?)
            } else {
                None
            };

            let name = if self.match_token(&TokenType::As) {
                Some(self.expect_identifier()?)
            } else {
                None
            };

            self.expect(TokenType::Colon)?;
            let handler_body = self.block()?;

            handlers.push(ExceptHandler {
                exception_type,
                name,
                body: handler_body,
            });
        }

        let else_body = if self.match_token(&TokenType::Else) {
            self.expect(TokenType::Colon)?;
            self.block()?
        } else {
            Vec::new()
        };

        let finally_body = if self.match_token(&TokenType::Finally) {
            self.expect(TokenType::Colon)?;
            self.block()?
        } else {
            Vec::new()
        };

        Ok(Stmt::Try {
            body,
            handlers,
            else_body,
            finally_body,
        })
    }

    fn return_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenType::Return)?;
        let value = if !self.check(&TokenType::Newline) && !self.is_at_end() {
            Some(self.expression()?)
        } else {
            None
        };
        self.expect_newline()?;
        Ok(Stmt::Return(value))
    }

    fn raise_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenType::Raise)?;
        let value = if !self.check(&TokenType::Newline) && !self.is_at_end() {
            Some(self.expression()?)
        } else {
            None
        };
        self.expect_newline()?;
        Ok(Stmt::Raise(value))
    }

    fn global_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenType::Global)?;
        let mut names = vec![self.expect_identifier()?];
        while self.match_token(&TokenType::Comma) {
            names.push(self.expect_identifier()?);
        }
        self.expect_newline()?;
        Ok(Stmt::Global(names))
    }

    fn nonlocal_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenType::Nonlocal)?;
        let mut names = vec![self.expect_identifier()?];
        while self.match_token(&TokenType::Comma) {
            names.push(self.expect_identifier()?);
        }
        self.expect_newline()?;
        Ok(Stmt::Nonlocal(names))
    }

    fn import_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenType::Import)?;
        let mut names = Vec::new();
        loop {
            let name = self.expect_identifier()?;
            let asname = if self.match_token(&TokenType::As) {
                Some(self.expect_identifier()?)
            } else {
                None
            };
            names.push(Alias { name, asname });
            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }
        self.expect_newline()?;
        Ok(Stmt::Import(names))
    }

    fn from_import_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenType::From)?;
        let module = self.expect_identifier()?;
        self.expect(TokenType::Import)?;

        let mut names = Vec::new();
        loop {
            let name = self.expect_identifier()?;
            let asname = if self.match_token(&TokenType::As) {
                Some(self.expect_identifier()?)
            } else {
                None
            };
            names.push(Alias { name, asname });
            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }
        self.expect_newline()?;
        Ok(Stmt::ImportFrom { module, names })
    }

    fn expr_or_assign_stmt(&mut self) -> Result<Stmt> {
        let expr = self.expression()?;

        // Check for assignment
        if self.match_token(&TokenType::Assign) {
            let value = self.expression()?;
            self.expect_newline()?;
            return Ok(Stmt::Assign {
                targets: vec![expr],
                value,
            });
        }

        // Check for augmented assignment
        let op = match self.peek().token_type {
            TokenType::PlusAssign => Some(BinaryOp::Add),
            TokenType::MinusAssign => Some(BinaryOp::Sub),
            TokenType::StarAssign => Some(BinaryOp::Mul),
            TokenType::SlashAssign => Some(BinaryOp::Div),
            TokenType::PercentAssign => Some(BinaryOp::Mod),
            TokenType::DoubleStarAssign => Some(BinaryOp::Pow),
            TokenType::DoubleSlashAssign => Some(BinaryOp::FloorDiv),
            _ => None,
        };

        if let Some(op) = op {
            self.advance();
            let value = self.expression()?;
            self.expect_newline()?;
            return Ok(Stmt::AugAssign {
                target: expr,
                op,
                value,
            });
        }

        self.expect_newline()?;
        Ok(Stmt::Expr(expr))
    }

    fn block(&mut self) -> Result<Vec<Stmt>> {
        self.expect_newline()?;
        self.expect(TokenType::Indent)?;

        let mut stmts = Vec::new();
        while !self.check(&TokenType::Dedent) && !self.is_at_end() {
            self.skip_newlines();
            if self.check(&TokenType::Dedent) {
                break;
            }
            stmts.push(self.statement()?);
        }

        if !self.is_at_end() {
            self.expect(TokenType::Dedent)?;
        }

        Ok(stmts)
    }

    // Expression parsing using Pratt parser
    fn expression(&mut self) -> Result<Expr> {
        self.parse_precedence(0, true)
    }

    /// Parse expression without handling ternary conditional (for list comprehension contexts).
    fn expression_no_cond(&mut self) -> Result<Expr> {
        self.parse_precedence(0, false)
    }

    fn parse_precedence(&mut self, min_prec: u8, allow_conditional: bool) -> Result<Expr> {
        let mut left = self.unary()?;

        while let Some(op) = self.peek_binary_op() {
            let prec = op.precedence();
            if prec < min_prec {
                break;
            }

            self.advance();
            let right = self.parse_precedence(
                if op.is_right_associative() { prec } else { prec + 1 },
                allow_conditional,
            )?;

            left = Expr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        // Handle comparison chains - only when precedence allows
        // Comparison operators have precedence ~3, so only handle when min_prec <= 3
        if min_prec <= 3 {
            if let Some(_op) = self.peek_compare_op() {
                let mut ops = Vec::new();
                while let Some(cmp_op) = self.peek_compare_op() {
                    self.advance();
                    let right = self.parse_precedence(4, allow_conditional)?;
                    ops.push((cmp_op, right));
                }
                left = Expr::Compare {
                    left: Box::new(left),
                    ops,
                };
            }
        }

        // Handle conditional expression (ternary) - only if allowed
        if allow_conditional && self.match_token(&TokenType::If) {
            let test = self.expression()?;
            self.expect(TokenType::Else)?;
            let orelse = self.expression()?;
            left = Expr::IfExpr {
                test: Box::new(test),
                body: Box::new(left),
                orelse: Box::new(orelse),
            };
        }

        Ok(left)
    }

    fn unary(&mut self) -> Result<Expr> {
        if self.match_token(&TokenType::Minus) {
            let operand = self.unary()?;
            return Ok(Expr::UnaryOp {
                op: UnaryOp::Neg,
                operand: Box::new(operand),
            });
        }

        if self.match_token(&TokenType::Not) {
            let operand = self.unary()?;
            return Ok(Expr::UnaryOp {
                op: UnaryOp::Not,
                operand: Box::new(operand),
            });
        }

        // yield [from] expression
        if self.match_token(&TokenType::Yield) {
            if self.match_token(&TokenType::From) {
                let expr = self.expression()?;
                return Ok(Expr::YieldFrom(Box::new(expr)));
            }
            let value = if !self.check(&TokenType::Newline)
                && !self.check(&TokenType::RParen)
                && !self.check(&TokenType::Comma)
                && !self.is_at_end()
            {
                Some(Box::new(self.expression()?))
            } else {
                None
            };
            return Ok(Expr::Yield(value));
        }

        // await expression
        if self.match_token(&TokenType::Await) {
            let expr = self.unary()?;
            return Ok(Expr::Await(Box::new(expr)));
        }

        self.call()
    }

    fn call(&mut self) -> Result<Expr> {
        let mut expr = self.primary()?;

        loop {
            if self.match_token(&TokenType::LParen) {
                let mut args = Vec::new();
                let kwargs = Vec::new();

                if !self.check(&TokenType::RParen) {
                    loop {
                        args.push(self.expression()?);
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }

                self.expect(TokenType::RParen)?;
                expr = Expr::Call {
                    func: Box::new(expr),
                    args,
                    kwargs,
                };
            } else if self.match_token(&TokenType::LBracket) {
                let index = self.expression()?;
                self.expect(TokenType::RBracket)?;
                expr = Expr::Subscript {
                    value: Box::new(expr),
                    index: Box::new(index),
                };
            } else if self.match_token(&TokenType::Dot) {
                let attr = self.expect_identifier()?;
                expr = Expr::Attribute {
                    value: Box::new(expr),
                    attr,
                };
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn primary(&mut self) -> Result<Expr> {
        let token = self.advance();

        match token.token_type {
            TokenType::Integer(n) => Ok(Expr::Integer(n)),
            TokenType::Float(n) => Ok(Expr::Float(n)),
            TokenType::String(s) => Ok(Expr::String(s)),
            TokenType::True => Ok(Expr::Bool(true)),
            TokenType::False => Ok(Expr::Bool(false)),
            TokenType::None => Ok(Expr::None),
            TokenType::Identifier(name) => Ok(Expr::Identifier(name)),

            TokenType::LParen => {
                let expr = self.expression()?;
                if self.match_token(&TokenType::Comma) {
                    // Tuple
                    let mut elements = vec![expr];
                    if !self.check(&TokenType::RParen) {
                        loop {
                            elements.push(self.expression()?);
                            if !self.match_token(&TokenType::Comma) {
                                break;
                            }
                        }
                    }
                    self.expect(TokenType::RParen)?;
                    Ok(Expr::Tuple(elements))
                } else {
                    self.expect(TokenType::RParen)?;
                    Ok(expr)
                }
            }

            TokenType::LBracket => {
                let mut elements = Vec::new();
                if !self.check(&TokenType::RBracket) {
                    let first = self.expression()?;

                    // Check for list comprehension
                    if self.match_token(&TokenType::For) {
                        let target = self.expect_identifier()?;
                        self.expect(TokenType::In)?;
                        // Use expression_no_cond to avoid consuming `if` as ternary
                        let iter = self.expression_no_cond()?;
                        let condition = if self.match_token(&TokenType::If) {
                            Some(Box::new(self.expression_no_cond()?))
                        } else {
                            None
                        };
                        self.expect(TokenType::RBracket)?;
                        return Ok(Expr::ListComp {
                            element: Box::new(first),
                            target,
                            iter: Box::new(iter),
                            condition,
                        });
                    }

                    elements.push(first);
                    while self.match_token(&TokenType::Comma) {
                        if self.check(&TokenType::RBracket) {
                            break;
                        }
                        elements.push(self.expression()?);
                    }
                }
                self.expect(TokenType::RBracket)?;
                Ok(Expr::List(elements))
            }

            TokenType::LBrace => {
                let mut pairs = Vec::new();
                if !self.check(&TokenType::RBrace) {
                    loop {
                        let key = self.expression()?;
                        self.expect(TokenType::Colon)?;
                        let value = self.expression()?;
                        pairs.push((key, value));
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                self.expect(TokenType::RBrace)?;
                Ok(Expr::Dict(pairs))
            }

            TokenType::Lambda => {
                let mut params = Vec::new();
                if !self.check(&TokenType::Colon) {
                    loop {
                        let name = self.expect_identifier()?;
                        params.push(Param {
                            name,
                            default: None,
                        });
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                self.expect(TokenType::Colon)?;
                let body = self.expression()?;
                Ok(Expr::Lambda {
                    params,
                    body: Box::new(body),
                })
            }

            _ => Err(self.error(&format!("Unexpected token: {:?}", token.token_type))),
        }
    }

    fn peek_binary_op(&self) -> Option<BinaryOp> {
        match self.peek().token_type {
            TokenType::Plus => Some(BinaryOp::Add),
            TokenType::Minus => Some(BinaryOp::Sub),
            TokenType::Star => Some(BinaryOp::Mul),
            TokenType::Slash => Some(BinaryOp::Div),
            TokenType::DoubleSlash => Some(BinaryOp::FloorDiv),
            TokenType::Percent => Some(BinaryOp::Mod),
            TokenType::DoubleStar => Some(BinaryOp::Pow),
            TokenType::And => Some(BinaryOp::And),
            TokenType::Or => Some(BinaryOp::Or),
            _ => None,
        }
    }

    fn peek_compare_op(&self) -> Option<CompareOp> {
        match self.peek().token_type {
            TokenType::Eq => Some(CompareOp::Eq),
            TokenType::Ne => Some(CompareOp::Ne),
            TokenType::Lt => Some(CompareOp::Lt),
            TokenType::Le => Some(CompareOp::Le),
            TokenType::Gt => Some(CompareOp::Gt),
            TokenType::Ge => Some(CompareOp::Ge),
            TokenType::Is => {
                if self.peek_next_type() == Some(TokenType::Not) {
                    Some(CompareOp::IsNot)
                } else {
                    Some(CompareOp::Is)
                }
            }
            TokenType::In => Some(CompareOp::In),
            TokenType::Not => {
                if self.peek_next_type() == Some(TokenType::In) {
                    Some(CompareOp::NotIn)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    // Helper methods
    fn advance(&mut self) -> Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.tokens[self.current - 1].clone()
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn peek_next_type(&self) -> Option<TokenType> {
        if self.current + 1 < self.tokens.len() {
            Some(self.tokens[self.current + 1].token_type.clone())
        } else {
            None
        }
    }

    fn check(&self, token_type: &TokenType) -> bool {
        std::mem::discriminant(&self.peek().token_type) == std::mem::discriminant(token_type)
    }

    fn match_token(&mut self, token_type: &TokenType) -> bool {
        if self.check(token_type) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, token_type: TokenType) -> Result<Token> {
        if self.check(&token_type) {
            Ok(self.advance())
        } else {
            Err(self.error(&format!(
                "Expected {:?}, found {:?}",
                token_type,
                self.peek().token_type
            )))
        }
    }

    fn expect_identifier(&mut self) -> Result<String> {
        match self.peek().token_type.clone() {
            TokenType::Identifier(name) => {
                self.advance();
                Ok(name)
            }
            _ => Err(self.error("Expected identifier")),
        }
    }

    fn expect_newline(&mut self) -> Result<()> {
        if self.is_at_end() {
            return Ok(());
        }
        if self.match_token(&TokenType::Newline) {
            Ok(())
        } else if self.check(&TokenType::Eof) || self.check(&TokenType::Dedent) {
            // Dedent is acceptable at end of block (no newline needed before dedent)
            Ok(())
        } else {
            Err(self.error("Expected newline"))
        }
    }

    fn skip_newlines(&mut self) {
        while self.match_token(&TokenType::Newline) {}
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek().token_type, TokenType::Eof)
    }

    fn error(&self, message: &str) -> Error {
        Error::Parse {
            message: message.to_string(),
            line: self.peek().line(),
        }
    }
}
