//! Comprehensive tests for the lexer.

use py_compiler::lexer::Lexer;
use py_compiler::token::TokenType;

/// Helper to get token types from source code.
fn tokenize(source: &str) -> Vec<TokenType> {
    Lexer::new(source)
        .tokenize()
        .unwrap()
        .into_iter()
        .map(|t| t.token_type)
        .collect()
}

// ============================================================================
// Basic Token Tests
// ============================================================================

#[test]
fn test_empty_source() {
    let tokens = tokenize("");
    assert_eq!(tokens, vec![TokenType::Eof]);
}

#[test]
fn test_whitespace_only() {
    let tokens = tokenize("   \t  ");
    assert_eq!(tokens, vec![TokenType::Eof]);
}

#[test]
fn test_simple_assignment() {
    let tokens = tokenize("x = 42");
    assert_eq!(
        tokens,
        vec![
            TokenType::Identifier("x".into()),
            TokenType::Assign,
            TokenType::Integer(42),
            TokenType::Eof,
        ]
    );
}

// ============================================================================
// Number Literal Tests
// ============================================================================

#[test]
fn test_integer_literals() {
    let tokens = tokenize("0 1 42 999999");
    assert_eq!(
        tokens,
        vec![
            TokenType::Integer(0),
            TokenType::Integer(1),
            TokenType::Integer(42),
            TokenType::Integer(999999),
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_float_literals() {
    let tokens = tokenize("3.14 0.5 2.0 .5");
    // Note: .5 should be handled as Dot followed by Integer
    assert!(tokens.contains(&TokenType::Float(3.14)));
    assert!(tokens.contains(&TokenType::Float(0.5)));
    assert!(tokens.contains(&TokenType::Float(2.0)));
}

#[test]
fn test_scientific_notation() {
    let tokens = tokenize("1e10 2.5e-3 1E5");
    assert!(tokens.contains(&TokenType::Float(1e10)));
    assert!(tokens.contains(&TokenType::Float(2.5e-3)));
    assert!(tokens.contains(&TokenType::Float(1e5)));
}

// ============================================================================
// String Literal Tests
// ============================================================================

#[test]
fn test_double_quoted_string() {
    let tokens = tokenize(r#""hello""#);
    assert_eq!(
        tokens,
        vec![TokenType::String("hello".into()), TokenType::Eof,]
    );
}

#[test]
fn test_single_quoted_string() {
    let tokens = tokenize("'world'");
    assert_eq!(
        tokens,
        vec![TokenType::String("world".into()), TokenType::Eof,]
    );
}

#[test]
fn test_string_escape_sequences() {
    let tokens = tokenize(r#""line1\nline2\ttab\\backslash""#);
    assert_eq!(
        tokens,
        vec![
            TokenType::String("line1\nline2\ttab\\backslash".into()),
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_string_with_escaped_quotes() {
    let tokens = tokenize(r#""say \"hello\"""#);
    assert_eq!(
        tokens,
        vec![TokenType::String("say \"hello\"".into()), TokenType::Eof,]
    );
}

#[test]
fn test_empty_string() {
    let tokens = tokenize(r#""""#);
    assert_eq!(
        tokens,
        vec![TokenType::String("".into()), TokenType::Eof,]
    );
}

// ============================================================================
// Operator Tests
// ============================================================================

#[test]
fn test_arithmetic_operators() {
    let tokens = tokenize("+ - * / // % **");
    assert_eq!(
        tokens,
        vec![
            TokenType::Plus,
            TokenType::Minus,
            TokenType::Star,
            TokenType::Slash,
            TokenType::DoubleSlash,
            TokenType::Percent,
            TokenType::DoubleStar,
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_comparison_operators() {
    let tokens = tokenize("== != < <= > >=");
    assert_eq!(
        tokens,
        vec![
            TokenType::Eq,
            TokenType::Ne,
            TokenType::Lt,
            TokenType::Le,
            TokenType::Gt,
            TokenType::Ge,
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_assignment_operators() {
    let tokens = tokenize("= += -= *= /= %= **= //=");
    assert_eq!(
        tokens,
        vec![
            TokenType::Assign,
            TokenType::PlusAssign,
            TokenType::MinusAssign,
            TokenType::StarAssign,
            TokenType::SlashAssign,
            TokenType::PercentAssign,
            TokenType::DoubleStarAssign,
            TokenType::DoubleSlashAssign,
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_arrow_operator() {
    let tokens = tokenize("->");
    assert_eq!(tokens, vec![TokenType::Arrow, TokenType::Eof,]);
}

// ============================================================================
// Delimiter Tests
// ============================================================================

#[test]
fn test_parentheses() {
    let tokens = tokenize("()");
    assert_eq!(
        tokens,
        vec![TokenType::LParen, TokenType::RParen, TokenType::Eof,]
    );
}

#[test]
fn test_brackets() {
    let tokens = tokenize("[]");
    assert_eq!(
        tokens,
        vec![TokenType::LBracket, TokenType::RBracket, TokenType::Eof,]
    );
}

#[test]
fn test_braces() {
    let tokens = tokenize("{}");
    assert_eq!(
        tokens,
        vec![TokenType::LBrace, TokenType::RBrace, TokenType::Eof,]
    );
}

#[test]
fn test_punctuation() {
    let tokens = tokenize(",:;.");
    assert_eq!(
        tokens,
        vec![
            TokenType::Comma,
            TokenType::Colon,
            TokenType::Semicolon,
            TokenType::Dot,
            TokenType::Eof,
        ]
    );
}

// ============================================================================
// Keyword Tests
// ============================================================================

#[test]
fn test_control_flow_keywords() {
    let tokens = tokenize("if elif else while for in");
    assert_eq!(
        tokens,
        vec![
            TokenType::If,
            TokenType::Elif,
            TokenType::Else,
            TokenType::While,
            TokenType::For,
            TokenType::In,
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_function_keywords() {
    let tokens = tokenize("def return lambda");
    assert_eq!(
        tokens,
        vec![
            TokenType::Def,
            TokenType::Return,
            TokenType::Lambda,
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_class_keywords() {
    let tokens = tokenize("class");
    assert_eq!(tokens, vec![TokenType::Class, TokenType::Eof,]);
}

#[test]
fn test_exception_keywords() {
    let tokens = tokenize("try except finally raise");
    assert_eq!(
        tokens,
        vec![
            TokenType::Try,
            TokenType::Except,
            TokenType::Finally,
            TokenType::Raise,
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_import_keywords() {
    let tokens = tokenize("import from as");
    assert_eq!(
        tokens,
        vec![
            TokenType::Import,
            TokenType::From,
            TokenType::As,
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_logical_keywords() {
    let tokens = tokenize("and or not is");
    assert_eq!(
        tokens,
        vec![
            TokenType::And,
            TokenType::Or,
            TokenType::Not,
            TokenType::Is,
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_scope_keywords() {
    let tokens = tokenize("global nonlocal");
    assert_eq!(
        tokens,
        vec![TokenType::Global, TokenType::Nonlocal, TokenType::Eof,]
    );
}

#[test]
fn test_control_keywords() {
    let tokens = tokenize("pass break continue");
    assert_eq!(
        tokens,
        vec![
            TokenType::Pass,
            TokenType::Break,
            TokenType::Continue,
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_boolean_and_none_literals() {
    let tokens = tokenize("True False None");
    assert_eq!(
        tokens,
        vec![
            TokenType::True,
            TokenType::False,
            TokenType::None,
            TokenType::Eof,
        ]
    );
}

// ============================================================================
// Identifier Tests
// ============================================================================

#[test]
fn test_simple_identifiers() {
    let tokens = tokenize("foo bar baz");
    assert_eq!(
        tokens,
        vec![
            TokenType::Identifier("foo".into()),
            TokenType::Identifier("bar".into()),
            TokenType::Identifier("baz".into()),
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_underscore_identifiers() {
    let tokens = tokenize("_private __dunder__ _");
    assert_eq!(
        tokens,
        vec![
            TokenType::Identifier("_private".into()),
            TokenType::Identifier("__dunder__".into()),
            TokenType::Identifier("_".into()),
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_identifier_with_numbers() {
    let tokens = tokenize("var1 my2nd thing3");
    assert_eq!(
        tokens,
        vec![
            TokenType::Identifier("var1".into()),
            TokenType::Identifier("my2nd".into()),
            TokenType::Identifier("thing3".into()),
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_keyword_like_identifiers() {
    // These should be identifiers, not keywords
    let tokens = tokenize("iffy define classy");
    assert_eq!(
        tokens,
        vec![
            TokenType::Identifier("iffy".into()),
            TokenType::Identifier("define".into()),
            TokenType::Identifier("classy".into()),
            TokenType::Eof,
        ]
    );
}

// ============================================================================
// Comment Tests
// ============================================================================

#[test]
fn test_comment_at_end_of_line() {
    let tokens = tokenize("x = 1 # this is a comment");
    assert_eq!(
        tokens,
        vec![
            TokenType::Identifier("x".into()),
            TokenType::Assign,
            TokenType::Integer(1),
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_comment_only_line() {
    let tokens = tokenize("# just a comment");
    assert_eq!(tokens, vec![TokenType::Eof]);
}

#[test]
fn test_code_after_comment_line() {
    let tokens = tokenize("# comment\nx = 1");
    assert_eq!(
        tokens,
        vec![
            TokenType::Newline,
            TokenType::Identifier("x".into()),
            TokenType::Assign,
            TokenType::Integer(1),
            TokenType::Eof,
        ]
    );
}

// ============================================================================
// Indentation Tests
// ============================================================================

#[test]
fn test_simple_indentation() {
    let tokens = tokenize("if x:\n    y");
    assert!(tokens.contains(&TokenType::Indent));
    assert!(tokens.contains(&TokenType::Dedent));
}

#[test]
fn test_multiple_indent_levels() {
    let source = "if x:\n    if y:\n        z\n    w";
    let tokens = tokenize(source);
    let indent_count = tokens.iter().filter(|t| **t == TokenType::Indent).count();
    let dedent_count = tokens.iter().filter(|t| **t == TokenType::Dedent).count();
    assert_eq!(indent_count, 2);
    assert_eq!(dedent_count, 2);
}

#[test]
fn test_dedent_to_zero() {
    let source = "if x:\n    y\nz";
    let tokens = tokenize(source);
    assert!(tokens.contains(&TokenType::Indent));
    assert!(tokens.contains(&TokenType::Dedent));
}

#[test]
fn test_newlines_without_indent() {
    let tokens = tokenize("x\ny\nz");
    let newline_count = tokens.iter().filter(|t| **t == TokenType::Newline).count();
    assert_eq!(newline_count, 2);
}

// ============================================================================
// Parenthesis Continuation Tests
// ============================================================================

#[test]
fn test_no_newline_inside_parens() {
    let tokens = tokenize("(1 +\n2)");
    // Newlines inside parentheses should be ignored
    let newline_count = tokens.iter().filter(|t| **t == TokenType::Newline).count();
    assert_eq!(newline_count, 0);
}

#[test]
fn test_no_newline_inside_brackets() {
    let tokens = tokenize("[1,\n2]");
    let newline_count = tokens.iter().filter(|t| **t == TokenType::Newline).count();
    assert_eq!(newline_count, 0);
}

#[test]
fn test_no_newline_inside_braces() {
    let tokens = tokenize("{1:\n2}");
    let newline_count = tokens.iter().filter(|t| **t == TokenType::Newline).count();
    assert_eq!(newline_count, 0);
}

// ============================================================================
// Complex Expression Tests
// ============================================================================

#[test]
fn test_function_call_expression() {
    let tokens = tokenize("print(x, y)");
    assert_eq!(
        tokens,
        vec![
            TokenType::Identifier("print".into()),
            TokenType::LParen,
            TokenType::Identifier("x".into()),
            TokenType::Comma,
            TokenType::Identifier("y".into()),
            TokenType::RParen,
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_list_literal() {
    let tokens = tokenize("[1, 2, 3]");
    assert_eq!(
        tokens,
        vec![
            TokenType::LBracket,
            TokenType::Integer(1),
            TokenType::Comma,
            TokenType::Integer(2),
            TokenType::Comma,
            TokenType::Integer(3),
            TokenType::RBracket,
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_dict_literal() {
    let tokens = tokenize(r#"{"a": 1}"#);
    assert_eq!(
        tokens,
        vec![
            TokenType::LBrace,
            TokenType::String("a".into()),
            TokenType::Colon,
            TokenType::Integer(1),
            TokenType::RBrace,
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_attribute_access() {
    let tokens = tokenize("obj.attr");
    assert_eq!(
        tokens,
        vec![
            TokenType::Identifier("obj".into()),
            TokenType::Dot,
            TokenType::Identifier("attr".into()),
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_subscript() {
    let tokens = tokenize("arr[0]");
    assert_eq!(
        tokens,
        vec![
            TokenType::Identifier("arr".into()),
            TokenType::LBracket,
            TokenType::Integer(0),
            TokenType::RBracket,
            TokenType::Eof,
        ]
    );
}

// ============================================================================
// Function Definition Tokens
// ============================================================================

#[test]
fn test_function_definition_tokens() {
    let source = "def add(a, b):\n    return a + b";
    let tokens = tokenize(source);

    assert!(tokens.contains(&TokenType::Def));
    assert!(tokens.contains(&TokenType::Identifier("add".into())));
    assert!(tokens.contains(&TokenType::LParen));
    assert!(tokens.contains(&TokenType::Identifier("a".into())));
    assert!(tokens.contains(&TokenType::Comma));
    assert!(tokens.contains(&TokenType::Identifier("b".into())));
    assert!(tokens.contains(&TokenType::RParen));
    assert!(tokens.contains(&TokenType::Colon));
    assert!(tokens.contains(&TokenType::Return));
    assert!(tokens.contains(&TokenType::Plus));
}

// ============================================================================
// Class Definition Tokens
// ============================================================================

#[test]
fn test_class_definition_tokens() {
    let source = "class Point(Base):\n    pass";
    let tokens = tokenize(source);

    assert!(tokens.contains(&TokenType::Class));
    assert!(tokens.contains(&TokenType::Identifier("Point".into())));
    assert!(tokens.contains(&TokenType::LParen));
    assert!(tokens.contains(&TokenType::Identifier("Base".into())));
    assert!(tokens.contains(&TokenType::RParen));
    assert!(tokens.contains(&TokenType::Colon));
    assert!(tokens.contains(&TokenType::Pass));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_unterminated_string_error() {
    let result = Lexer::new(r#""unterminated"#).tokenize();
    assert!(result.is_err());
}

#[test]
fn test_unexpected_character_error() {
    let result = Lexer::new("x = $").tokenize();
    assert!(result.is_err());
}

#[test]
fn test_lone_exclamation_error() {
    let result = Lexer::new("!").tokenize();
    assert!(result.is_err());
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_at_operator() {
    let tokens = tokenize("@decorator");
    assert!(tokens.contains(&TokenType::At));
}

#[test]
fn test_chained_comparison_tokens() {
    let tokens = tokenize("1 < x < 10");
    assert_eq!(
        tokens,
        vec![
            TokenType::Integer(1),
            TokenType::Lt,
            TokenType::Identifier("x".into()),
            TokenType::Lt,
            TokenType::Integer(10),
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_negative_number_as_unary() {
    // -1 should be Minus followed by Integer
    let tokens = tokenize("-1");
    assert_eq!(
        tokens,
        vec![TokenType::Minus, TokenType::Integer(1), TokenType::Eof,]
    );
}

#[test]
fn test_multiline_expression() {
    let source = "x = (1 +\n     2 +\n     3)";
    let tokens = tokenize(source);
    // Should have no newlines because inside parens
    let newline_count = tokens.iter().filter(|t| **t == TokenType::Newline).count();
    assert_eq!(newline_count, 0);
    // Should have the final values
    assert!(tokens.contains(&TokenType::Integer(1)));
    assert!(tokens.contains(&TokenType::Integer(2)));
    assert!(tokens.contains(&TokenType::Integer(3)));
}

#[test]
fn test_power_operator_precedence() {
    let tokens = tokenize("2**3");
    assert_eq!(
        tokens,
        vec![
            TokenType::Integer(2),
            TokenType::DoubleStar,
            TokenType::Integer(3),
            TokenType::Eof,
        ]
    );
}

#[test]
fn test_floor_division_assignment() {
    let tokens = tokenize("x //= 2");
    assert_eq!(
        tokens,
        vec![
            TokenType::Identifier("x".into()),
            TokenType::DoubleSlashAssign,
            TokenType::Integer(2),
            TokenType::Eof,
        ]
    );
}
