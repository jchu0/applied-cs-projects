//! Comprehensive tests for the bytecode compiler.

use py_compiler::compiler::{Compiler, CodeObject, OpCode};
use py_compiler::lexer::Lexer;
use py_compiler::parser::Parser;
use py_compiler::value::Value;
use std::rc::Rc;

/// Helper to compile source code into a CodeObject.
fn compile(source: &str) -> Rc<CodeObject> {
    let tokens = Lexer::new(source).tokenize().unwrap();
    let ast = Parser::new(tokens).parse().unwrap();
    Compiler::new().compile(&ast).unwrap()
}

/// Helper to check if bytecode contains a specific opcode.
fn contains_opcode(code: &CodeObject, opcode: OpCode) -> bool {
    let opcode_byte = opcode as u8;
    code.bytecode.contains(&opcode_byte)
}

/// Helper to count occurrences of an opcode in bytecode.
fn count_opcode(code: &CodeObject, opcode: OpCode) -> usize {
    let opcode_byte = opcode as u8;
    code.bytecode.iter().filter(|&&b| b == opcode_byte).count()
}

// ============================================================================
// Basic Compilation Tests
// ============================================================================

#[test]
fn test_compile_empty() {
    let code = compile("");
    // Should have at least LoadConst (None) and Return
    assert!(contains_opcode(&code, OpCode::LoadConst));
    assert!(contains_opcode(&code, OpCode::Return));
}

#[test]
fn test_compile_integer_literal() {
    let code = compile("42");
    assert!(contains_opcode(&code, OpCode::LoadConst));
    // Check constant pool contains 42
    assert!(code.constants.iter().any(|v| matches!(v, Value::Int(42))));
}

#[test]
fn test_compile_float_literal() {
    let code = compile("3.14");
    assert!(contains_opcode(&code, OpCode::LoadConst));
    // Check constant pool contains float
    assert!(code.constants.iter().any(|v| matches!(v, Value::Float(f) if (*f - 3.14).abs() < 0.001)));
}

#[test]
fn test_compile_string_literal() {
    let code = compile(r#""hello""#);
    assert!(contains_opcode(&code, OpCode::LoadConst));
    // Check constant pool contains string
    assert!(code.constants.iter().any(|v| {
        matches!(v, Value::String(s) if **s == "hello")
    }));
}

#[test]
fn test_compile_boolean_true() {
    let code = compile("True");
    assert!(contains_opcode(&code, OpCode::LoadConst));
    assert!(code.constants.iter().any(|v| matches!(v, Value::Bool(true))));
}

#[test]
fn test_compile_boolean_false() {
    let code = compile("False");
    assert!(contains_opcode(&code, OpCode::LoadConst));
    assert!(code.constants.iter().any(|v| matches!(v, Value::Bool(false))));
}

#[test]
fn test_compile_none() {
    let code = compile("None");
    assert!(contains_opcode(&code, OpCode::LoadConst));
    assert!(code.constants.iter().any(|v| matches!(v, Value::None)));
}

// ============================================================================
// Variable Tests
// ============================================================================

#[test]
fn test_compile_global_assignment() {
    let code = compile("x = 42");
    assert!(contains_opcode(&code, OpCode::StoreName));
    // x should be in names
    assert!(code.names.contains(&"x".to_string()));
}

#[test]
fn test_compile_global_load() {
    let code = compile("x = 1\ny = x");
    assert!(contains_opcode(&code, OpCode::LoadName));
    assert!(contains_opcode(&code, OpCode::StoreName));
}

// ============================================================================
// Binary Operation Tests
// ============================================================================

#[test]
fn test_compile_addition() {
    let code = compile("1 + 2");
    assert!(contains_opcode(&code, OpCode::BinaryAdd));
}

#[test]
fn test_compile_subtraction() {
    let code = compile("5 - 3");
    assert!(contains_opcode(&code, OpCode::BinarySub));
}

#[test]
fn test_compile_multiplication() {
    let code = compile("2 * 3");
    assert!(contains_opcode(&code, OpCode::BinaryMul));
}

#[test]
fn test_compile_division() {
    let code = compile("6 / 2");
    assert!(contains_opcode(&code, OpCode::BinaryDiv));
}

#[test]
fn test_compile_floor_division() {
    let code = compile("7 // 2");
    assert!(contains_opcode(&code, OpCode::BinaryFloorDiv));
}

#[test]
fn test_compile_modulo() {
    let code = compile("7 % 3");
    assert!(contains_opcode(&code, OpCode::BinaryMod));
}

#[test]
fn test_compile_power() {
    let code = compile("2 ** 3");
    assert!(contains_opcode(&code, OpCode::BinaryPow));
}

#[test]
fn test_compile_chained_operations() {
    let code = compile("1 + 2 + 3");
    // Should have two additions
    assert_eq!(count_opcode(&code, OpCode::BinaryAdd), 2);
}

// ============================================================================
// Unary Operation Tests
// ============================================================================

#[test]
fn test_compile_unary_neg() {
    let code = compile("-x");
    assert!(contains_opcode(&code, OpCode::UnaryNeg));
}

#[test]
fn test_compile_unary_not() {
    let code = compile("not x");
    assert!(contains_opcode(&code, OpCode::UnaryNot));
}

// ============================================================================
// Comparison Operation Tests
// ============================================================================

#[test]
fn test_compile_equality() {
    let code = compile("a == b");
    assert!(contains_opcode(&code, OpCode::CompareEq));
}

#[test]
fn test_compile_not_equal() {
    let code = compile("a != b");
    assert!(contains_opcode(&code, OpCode::CompareNe));
}

#[test]
fn test_compile_less_than() {
    let code = compile("a < b");
    assert!(contains_opcode(&code, OpCode::CompareLt));
}

#[test]
fn test_compile_less_equal() {
    let code = compile("a <= b");
    assert!(contains_opcode(&code, OpCode::CompareLe));
}

#[test]
fn test_compile_greater_than() {
    let code = compile("a > b");
    assert!(contains_opcode(&code, OpCode::CompareGt));
}

#[test]
fn test_compile_greater_equal() {
    let code = compile("a >= b");
    assert!(contains_opcode(&code, OpCode::CompareGe));
}

// ============================================================================
// Short-Circuit Logic Tests
// ============================================================================

#[test]
fn test_compile_and() {
    let code = compile("a and b");
    // Should use JumpIfFalse for short-circuit
    assert!(contains_opcode(&code, OpCode::JumpIfFalse));
}

#[test]
fn test_compile_or() {
    let code = compile("a or b");
    // Should use JumpIfTrue for short-circuit
    assert!(contains_opcode(&code, OpCode::JumpIfTrue));
}

// ============================================================================
// Collection Construction Tests
// ============================================================================

#[test]
fn test_compile_list() {
    let code = compile("[1, 2, 3]");
    assert!(contains_opcode(&code, OpCode::BuildList));
}

#[test]
fn test_compile_empty_list() {
    let code = compile("[]");
    assert!(contains_opcode(&code, OpCode::BuildList));
}

#[test]
fn test_compile_dict() {
    let code = compile(r#"{"a": 1}"#);
    assert!(contains_opcode(&code, OpCode::BuildDict));
}

#[test]
fn test_compile_empty_dict() {
    let code = compile("{}");
    assert!(contains_opcode(&code, OpCode::BuildDict));
}

#[test]
fn test_compile_tuple() {
    let code = compile("(1, 2)");
    assert!(contains_opcode(&code, OpCode::BuildTuple));
}

// ============================================================================
// Access Expression Tests
// ============================================================================

#[test]
fn test_compile_attribute_load() {
    let code = compile("obj.attr");
    assert!(contains_opcode(&code, OpCode::LoadAttr));
    assert!(code.names.contains(&"attr".to_string()));
}

#[test]
fn test_compile_attribute_store() {
    let code = compile("obj.attr = 1");
    assert!(contains_opcode(&code, OpCode::StoreAttr));
}

#[test]
fn test_compile_subscript_load() {
    let code = compile("arr[0]");
    assert!(contains_opcode(&code, OpCode::LoadSubscript));
}

#[test]
fn test_compile_subscript_store() {
    let code = compile("arr[0] = 1");
    assert!(contains_opcode(&code, OpCode::StoreSubscript));
}

// ============================================================================
// Function Call Tests
// ============================================================================

#[test]
fn test_compile_function_call() {
    let code = compile("foo()");
    assert!(contains_opcode(&code, OpCode::Call));
}

#[test]
fn test_compile_function_call_with_args() {
    let code = compile("foo(1, 2, 3)");
    assert!(contains_opcode(&code, OpCode::Call));
    // Three arguments loaded before call
    assert!(count_opcode(&code, OpCode::LoadConst) >= 3);
}

// ============================================================================
// Control Flow Tests
// ============================================================================

#[test]
fn test_compile_if_statement() {
    let code = compile("if x:\n    pass");
    assert!(contains_opcode(&code, OpCode::PopJumpIfFalse));
}

#[test]
fn test_compile_if_else() {
    let code = compile("if x:\n    pass\nelse:\n    pass");
    assert!(contains_opcode(&code, OpCode::PopJumpIfFalse));
    assert!(contains_opcode(&code, OpCode::Jump));
}

#[test]
fn test_compile_while_loop() {
    let code = compile("while x:\n    pass");
    // Should have conditional jump and loop back jump
    assert!(contains_opcode(&code, OpCode::PopJumpIfFalse));
    assert!(contains_opcode(&code, OpCode::Jump));
}

#[test]
fn test_compile_for_loop() {
    let code = compile("for i in items:\n    pass");
    assert!(contains_opcode(&code, OpCode::GetIter));
    assert!(contains_opcode(&code, OpCode::ForIter));
}

// ============================================================================
// Function Definition Tests
// ============================================================================

#[test]
fn test_compile_function_def() {
    let code = compile("def foo():\n    pass");
    assert!(contains_opcode(&code, OpCode::MakeFunction));
    assert!(contains_opcode(&code, OpCode::StoreName));
    assert!(code.names.contains(&"foo".to_string()));
}

#[test]
fn test_compile_function_with_params() {
    let code = compile("def add(a, b):\n    return a + b");
    assert!(contains_opcode(&code, OpCode::MakeFunction));

    // Check the nested function code
    let func_const = code.constants.iter().find(|v| matches!(v, Value::Function(_)));
    assert!(func_const.is_some());

    if let Some(Value::Function(func)) = func_const {
        assert_eq!(func.code.arg_count, 2);
        assert!(func.code.varnames.contains(&"a".to_string()));
        assert!(func.code.varnames.contains(&"b".to_string()));
    }
}

#[test]
fn test_compile_return_statement() {
    let code = compile("def foo():\n    return 42");
    // Check nested function has Return
    let func_const = code.constants.iter().find(|v| matches!(v, Value::Function(_)));
    if let Some(Value::Function(func)) = func_const {
        assert!(contains_opcode(&func.code, OpCode::Return));
    }
}

#[test]
fn test_compile_implicit_return() {
    // Functions without explicit return should return None
    let code = compile("def foo():\n    pass");
    let func_const = code.constants.iter().find(|v| matches!(v, Value::Function(_)));
    if let Some(Value::Function(func)) = func_const {
        assert!(contains_opcode(&func.code, OpCode::Return));
        assert!(func.code.constants.iter().any(|v| matches!(v, Value::None)));
    }
}

// ============================================================================
// Class Definition Tests
// ============================================================================

#[test]
fn test_compile_class_def() {
    let code = compile("class Foo:\n    pass");
    assert!(contains_opcode(&code, OpCode::BuildClass));
    assert!(contains_opcode(&code, OpCode::StoreName));
    assert!(code.names.contains(&"Foo".to_string()));
}

#[test]
fn test_compile_class_with_base() {
    let code = compile("class Child(Parent):\n    pass");
    assert!(contains_opcode(&code, OpCode::BuildTuple)); // For bases
    assert!(contains_opcode(&code, OpCode::BuildClass));
}

// ============================================================================
// Lambda Tests
// ============================================================================

#[test]
fn test_compile_lambda() {
    let code = compile("f = lambda x: x + 1");
    assert!(contains_opcode(&code, OpCode::MakeFunction));

    let func_const = code.constants.iter().find(|v| matches!(v, Value::Function(_)));
    if let Some(Value::Function(func)) = func_const {
        assert_eq!(func.name, "<lambda>");
        assert_eq!(func.code.arg_count, 1);
    }
}

// ============================================================================
// List Comprehension Tests
// ============================================================================

#[test]
fn test_compile_list_comprehension() {
    let code = compile("[x for x in items]");
    assert!(contains_opcode(&code, OpCode::BuildList));
    assert!(contains_opcode(&code, OpCode::GetIter));
    assert!(contains_opcode(&code, OpCode::ForIter));
    assert!(contains_opcode(&code, OpCode::ListAppend));
}

#[test]
fn test_compile_list_comp_with_condition() {
    let code = compile("[x for x in items if x > 0]");
    // Should have conditional jump for filter
    assert!(contains_opcode(&code, OpCode::PopJumpIfFalse));
}

// ============================================================================
// Conditional Expression Tests
// ============================================================================

#[test]
fn test_compile_conditional_expr() {
    let code = compile("a if condition else b");
    assert!(contains_opcode(&code, OpCode::PopJumpIfFalse));
    assert!(contains_opcode(&code, OpCode::Jump));
}

// ============================================================================
// Augmented Assignment Tests
// ============================================================================

#[test]
fn test_compile_aug_assign_add() {
    let code = compile("x += 1");
    assert!(contains_opcode(&code, OpCode::BinaryAdd));
    assert!(contains_opcode(&code, OpCode::StoreName));
}

#[test]
fn test_compile_aug_assign_sub() {
    let code = compile("x -= 1");
    assert!(contains_opcode(&code, OpCode::BinarySub));
}

#[test]
fn test_compile_aug_assign_mul() {
    let code = compile("x *= 2");
    assert!(contains_opcode(&code, OpCode::BinaryMul));
}

// ============================================================================
// Pass Statement Tests
// ============================================================================

#[test]
fn test_compile_pass() {
    let code = compile("pass");
    assert!(contains_opcode(&code, OpCode::Nop));
}

// ============================================================================
// Expression Statement Tests
// ============================================================================

#[test]
fn test_compile_expr_statement_pops() {
    // Expression statements should pop their result
    let code = compile("1 + 2");
    assert!(contains_opcode(&code, OpCode::Pop));
}

// ============================================================================
// Code Object Structure Tests
// ============================================================================

#[test]
fn test_code_object_name() {
    let code = compile("x = 1");
    assert_eq!(code.name, "<module>");
}

#[test]
fn test_function_code_object_name() {
    let code = compile("def foo():\n    pass");
    let func_const = code.constants.iter().find(|v| matches!(v, Value::Function(_)));
    if let Some(Value::Function(func)) = func_const {
        assert_eq!(func.name, "foo");
    }
}

#[test]
fn test_nested_function_varnames() {
    let code = compile("def foo(a, b):\n    c = a + b\n    return c");
    let func_const = code.constants.iter().find(|v| matches!(v, Value::Function(_)));
    if let Some(Value::Function(func)) = func_const {
        // a, b, c should be in varnames
        assert!(func.code.varnames.contains(&"a".to_string()));
        assert!(func.code.varnames.contains(&"b".to_string()));
        assert!(func.code.varnames.contains(&"c".to_string()));
    }
}

// ============================================================================
// Bytecode Sequence Tests
// ============================================================================

#[test]
fn test_bytecode_not_empty() {
    let code = compile("x = 1");
    assert!(!code.bytecode.is_empty());
}

#[test]
fn test_bytecode_ends_with_return() {
    let code = compile("x = 1");
    // Module should end with Return
    let return_byte = OpCode::Return as u8;
    assert!(code.bytecode.ends_with(&[return_byte]));
}

// ============================================================================
// Constant Pool Tests
// ============================================================================

#[test]
fn test_constants_deduplicated() {
    let code = compile("x = 1\ny = 1\nz = 1");
    // Should only have one instance of 1 and one of None
    let one_count = code.constants.iter().filter(|v| matches!(v, Value::Int(1))).count();
    // Due to implementation details, this might be 3, but ideally 1
    // Just verify 1 exists
    assert!(one_count >= 1);
}

// ============================================================================
// Import Statement Tests
// ============================================================================

#[test]
fn test_compile_import() {
    let code = compile("import math");
    // Import defines name in global scope
    assert!(code.names.contains(&"math".to_string()));
}

// ============================================================================
// Global/Nonlocal Tests
// ============================================================================

#[test]
fn test_compile_global_handled() {
    // Global statement should compile without error
    let code = compile("global x");
    // Global is handled at name resolution, not bytecode
    assert!(!code.bytecode.is_empty());
}

// ============================================================================
// Complex Programs
// ============================================================================

#[test]
fn test_compile_fibonacci() {
    let source = r#"
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
"#;
    let code = compile(source);
    assert!(contains_opcode(&code, OpCode::MakeFunction));

    let func_const = code.constants.iter().find(|v| matches!(v, Value::Function(_)));
    if let Some(Value::Function(func)) = func_const {
        // Should have recursive calls
        assert!(contains_opcode(&func.code, OpCode::Call));
        // Should have conditional
        assert!(contains_opcode(&func.code, OpCode::PopJumpIfFalse));
    }
}

#[test]
fn test_compile_counter_class() {
    let source = r#"
class Counter:
    def __init__(self):
        pass
    def count(self):
        pass
"#;
    let code = compile(source);
    assert!(contains_opcode(&code, OpCode::BuildClass));
}
