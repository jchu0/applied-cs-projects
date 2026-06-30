//! Comprehensive tests for the VM execution.

use py_compiler::value::Value;
use py_compiler::run;
use std::rc::Rc;

/// Helper macro to run code and check for specific value.
fn run_code(source: &str) -> Value {
    run(source).expect("Code should execute successfully")
}

// ============================================================================
// Literal Evaluation Tests
// ============================================================================

#[test]
fn test_eval_integer() {
    let result = run_code("42");
    assert!(matches!(result, Value::None)); // Expressions don't return in module context
}

// ============================================================================
// Arithmetic Operation Tests
// ============================================================================

#[test]
fn test_arithmetic_addition() {
    let result = run_code("x = 1 + 2");
    // Assignment returns None at module level
    assert!(matches!(result, Value::None));
}

#[test]
fn test_arithmetic_subtraction() {
    // Verify subtraction works - run_code returns Value, not Result
    let result = run_code("x = 10 - 3");
    assert!(matches!(result, Value::None)); // Assignment returns None
}

// Use a different approach - test via the full pipeline

#[test]
fn test_run_returns_none_for_module() {
    // Module execution returns the last value which is None after return
    let result = run("x = 1");
    assert!(result.is_ok());
}

// ============================================================================
// Comprehensive End-to-End Tests Using run()
// ============================================================================

mod e2e_tests {
    use super::*;

    #[test]
    fn test_simple_assignment() {
        let result = run("x = 42");
        assert!(result.is_ok());
    }

    #[test]
    fn test_multiple_assignments() {
        let result = run("x = 1\ny = 2\nz = 3");
        assert!(result.is_ok());
    }

    #[test]
    fn test_binary_operations() {
        let ops = [
            "x = 1 + 2",
            "x = 5 - 3",
            "x = 2 * 3",
            "x = 6 / 2",
            "x = 7 // 2",
            "x = 7 % 3",
            "x = 2 ** 3",
        ];
        for op in ops {
            let result = run(op);
            assert!(result.is_ok(), "Failed for: {}", op);
        }
    }

    #[test]
    fn test_unary_operations() {
        assert!(run("x = -5").is_ok());
        assert!(run("x = not True").is_ok());
        assert!(run("x = not False").is_ok());
    }

    #[test]
    fn test_comparison_operations() {
        let ops = [
            "x = 1 == 1",
            "x = 1 != 2",
            "x = 1 < 2",
            "x = 2 <= 2",
            "x = 2 > 1",
            "x = 2 >= 2",
        ];
        for op in ops {
            let result = run(op);
            assert!(result.is_ok(), "Failed for: {}", op);
        }
    }

    #[test]
    fn test_boolean_logic() {
        assert!(run("x = True and True").is_ok());
        assert!(run("x = True and False").is_ok());
        assert!(run("x = True or False").is_ok());
        assert!(run("x = False or False").is_ok());
    }

    #[test]
    fn test_short_circuit_and() {
        // False and <anything> should not evaluate <anything>
        assert!(run("x = False and undefined_var").is_ok());
    }

    #[test]
    fn test_short_circuit_or() {
        // True or <anything> should not evaluate <anything>
        assert!(run("x = True or undefined_var").is_ok());
    }

    #[test]
    fn test_list_creation() {
        assert!(run("x = []").is_ok());
        assert!(run("x = [1]").is_ok());
        assert!(run("x = [1, 2, 3]").is_ok());
        assert!(run("x = [1, 2, 3, 4, 5]").is_ok());
    }

    #[test]
    fn test_dict_creation() {
        assert!(run("x = {}").is_ok());
        assert!(run(r#"x = {"a": 1}"#).is_ok());
        assert!(run(r#"x = {"a": 1, "b": 2}"#).is_ok());
    }

    #[test]
    fn test_tuple_creation() {
        assert!(run("x = (1, 2)").is_ok());
        assert!(run("x = (1, 2, 3)").is_ok());
    }

    #[test]
    fn test_list_subscript() {
        assert!(run("x = [1, 2, 3]\ny = x[0]").is_ok());
        assert!(run("x = [1, 2, 3]\ny = x[1]").is_ok());
        assert!(run("x = [1, 2, 3]\ny = x[-1]").is_ok());
    }

    #[test]
    fn test_list_assignment() {
        assert!(run("x = [1, 2, 3]\nx[0] = 10").is_ok());
    }

    #[test]
    fn test_dict_subscript() {
        let code = "x = {\"a\": 1}\ny = x[\"a\"]";
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_dict_assignment() {
        let code = "x = {\"a\": 1}\nx[\"b\"] = 2";
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_string_concatenation() {
        assert!(run(r#"x = "hello" + " " + "world""#).is_ok());
    }

    #[test]
    fn test_string_repetition() {
        assert!(run(r#"x = "ab" * 3"#).is_ok());
    }

    #[test]
    fn test_if_statement_true() {
        let code = r#"
x = 0
if True:
    x = 1
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_if_statement_false() {
        let code = r#"
x = 0
if False:
    x = 1
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_if_else() {
        let code = r#"
if False:
    x = 1
else:
    x = 2
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_if_elif_else() {
        let code = r#"
x = 2
if x == 1:
    y = 1
elif x == 2:
    y = 2
else:
    y = 3
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_while_loop() {
        let code = r#"
x = 0
i = 0
while i < 5:
    x = x + 1
    i = i + 1
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_for_loop_list() {
        let code = r#"
x = 0
for i in [1, 2, 3]:
    x = x + i
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_for_loop_range() {
        let code = r#"
x = 0
for i in range(5):
    x = x + i
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_nested_loops() {
        let code = r#"
x = 0
for i in range(3):
    for j in range(3):
        x = x + 1
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_function_definition() {
        let code = r#"
def foo():
    pass
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_function_call_no_args() {
        let code = r#"
def foo():
    pass
foo()
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_function_with_return() {
        let code = r#"
def add(a, b):
    return a + b
x = add(1, 2)
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_recursive_function() {
        let code = r#"
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
x = factorial(5)
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_fibonacci() {
        let code = r#"
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
x = fib(10)
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_lambda() {
        let code = r#"
f = lambda x: x + 1
y = f(5)
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_lambda_two_args() {
        let code = r#"
add = lambda a, b: a + b
x = add(3, 4)
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_class_definition() {
        let code = r#"
class Point:
    pass
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_class_instantiation() {
        let code = r#"
class Point:
    pass
p = Point()
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_class_with_init() {
        let code = r#"
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
p = Point(1, 2)
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_class_method_call() {
        let code = r#"
class Counter:
    def __init__(self):
        self.value = 0
    def increment(self):
        self.value = self.value + 1
c = Counter()
c.increment()
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_list_comprehension() {
        let code = r#"
x = [i for i in range(5)]
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_list_comp_with_expression() {
        let code = r#"
x = [i * 2 for i in range(5)]
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_list_comp_with_condition() {
        let code = r#"
x = [i for i in range(10) if i % 2 == 0]
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_conditional_expression() {
        assert!(run("x = 1 if True else 0").is_ok());
        assert!(run("x = 1 if False else 0").is_ok());
    }

    #[test]
    fn test_augmented_assignment() {
        let ops = [
            "x = 1\nx += 1",
            "x = 5\nx -= 1",
            "x = 2\nx *= 3",
            "x = 6\nx /= 2",
            "x = 7\nx //= 2",
            "x = 7\nx %= 3",
            "x = 2\nx **= 3",
        ];
        for op in ops {
            assert!(run(op).is_ok(), "Failed for: {}", op);
        }
    }

    #[test]
    fn test_pass_statement() {
        assert!(run("pass").is_ok());
    }

    #[test]
    fn test_multiple_functions() {
        let code = r#"
def foo():
    return 1

def bar():
    return 2

x = foo() + bar()
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_function_calling_function() {
        let code = r#"
def inner():
    return 42

def outer():
    return inner()

x = outer()
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_function_with_local_vars() {
        let code = r#"
def foo():
    x = 1
    y = 2
    return x + y
z = foo()
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_nested_class_method() {
        let code = r#"
class Outer:
    def method(self):
        return 42
o = Outer()
x = o.method()
"#;
        assert!(run(code).is_ok());
    }

    // ========================================================================
    // Built-in Function Tests
    // ========================================================================

    #[test]
    fn test_builtin_len_list() {
        assert!(run("x = len([1, 2, 3])").is_ok());
    }

    #[test]
    fn test_builtin_len_string() {
        assert!(run(r#"x = len("hello")"#).is_ok());
    }

    #[test]
    fn test_builtin_len_dict() {
        assert!(run(r#"x = len({"a": 1, "b": 2})"#).is_ok());
    }

    #[test]
    fn test_builtin_range_one_arg() {
        assert!(run("x = range(5)").is_ok());
    }

    #[test]
    fn test_builtin_range_two_args() {
        assert!(run("x = range(1, 5)").is_ok());
    }

    #[test]
    fn test_builtin_range_three_args() {
        assert!(run("x = range(0, 10, 2)").is_ok());
    }

    #[test]
    fn test_builtin_int_from_string() {
        assert!(run(r#"x = int("42")"#).is_ok());
    }

    #[test]
    fn test_builtin_int_from_float() {
        assert!(run("x = int(3.14)").is_ok());
    }

    #[test]
    fn test_builtin_float_from_int() {
        assert!(run("x = float(42)").is_ok());
    }

    #[test]
    fn test_builtin_str() {
        assert!(run("x = str(42)").is_ok());
    }

    #[test]
    fn test_builtin_bool() {
        assert!(run("x = bool(1)").is_ok());
        assert!(run("x = bool(0)").is_ok());
    }

    #[test]
    fn test_builtin_list_from_range() {
        assert!(run("x = list(range(5))").is_ok());
    }

    #[test]
    fn test_builtin_abs() {
        assert!(run("x = abs(-5)").is_ok());
        assert!(run("x = abs(5)").is_ok());
    }

    #[test]
    fn test_builtin_min() {
        assert!(run("x = min(1, 2, 3)").is_ok());
        assert!(run("x = min([1, 2, 3])").is_ok());
    }

    #[test]
    fn test_builtin_max() {
        assert!(run("x = max(1, 2, 3)").is_ok());
        assert!(run("x = max([1, 2, 3])").is_ok());
    }

    #[test]
    fn test_builtin_sum() {
        assert!(run("x = sum([1, 2, 3])").is_ok());
    }

    #[test]
    fn test_builtin_sorted() {
        assert!(run("x = sorted([3, 1, 2])").is_ok());
    }

    #[test]
    fn test_builtin_reversed() {
        assert!(run("x = reversed([1, 2, 3])").is_ok());
    }

    #[test]
    fn test_builtin_enumerate() {
        assert!(run("x = enumerate([1, 2, 3])").is_ok());
    }

    #[test]
    fn test_builtin_zip() {
        assert!(run("x = zip([1, 2], [3, 4])").is_ok());
    }

    #[test]
    fn test_builtin_type() {
        assert!(run("x = type(42)").is_ok());
        assert!(run(r#"x = type("hello")"#).is_ok());
    }

    #[test]
    fn test_builtin_round() {
        assert!(run("x = round(3.7)").is_ok());
        assert!(run("x = round(3.14159, 2)").is_ok());
    }

    #[test]
    fn test_builtin_ord() {
        assert!(run(r#"x = ord("A")"#).is_ok());
    }

    #[test]
    fn test_builtin_chr() {
        assert!(run("x = chr(65)").is_ok());
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    #[test]
    fn test_division_by_zero_int() {
        let result = run("x = 1 / 0");
        assert!(result.is_err());
    }

    #[test]
    fn test_division_by_zero_floor() {
        let result = run("x = 1 // 0");
        assert!(result.is_err());
    }

    #[test]
    fn test_modulo_by_zero() {
        let result = run("x = 1 % 0");
        assert!(result.is_err());
    }

    #[test]
    fn test_undefined_variable() {
        let result = run("x = undefined_var");
        assert!(result.is_err());
    }

    #[test]
    fn test_type_error_add() {
        let result = run(r#"x = 1 + "hello""#);
        assert!(result.is_err());
    }

    #[test]
    fn test_index_out_of_range() {
        let result = run("x = [1, 2, 3]\ny = x[10]");
        assert!(result.is_err());
    }

    #[test]
    fn test_key_error() {
        let result = run("x = {\"a\": 1}\ny = x[\"b\"]");
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_number_of_args() {
        let code = r#"
def add(a, b):
    return a + b
x = add(1)
"#;
        let result = run(code);
        assert!(result.is_err());
    }

    #[test]
    fn test_call_non_callable() {
        let result = run("x = 42\ny = x()");
        assert!(result.is_err());
    }

    // ========================================================================
    // Complex Program Tests
    // ========================================================================

    #[test]
    fn test_complex_arithmetic() {
        let code = r#"
x = (1 + 2) * 3 - 4 / 2
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_nested_functions() {
        let code = r#"
def outer(x):
    def inner(y):
        return y * 2
    return inner(x) + 1
z = outer(5)
"#;
        // Note: closures might not be fully implemented
        // This test verifies nested definitions work
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_object_oriented_example() {
        let code = r#"
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def area(self):
        return self.width * self.height
r = Rectangle(3, 4)
a = r.area()
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_iteration_with_processing() {
        let code = r#"
data = [1, 2, 3, 4, 5]
total = 0
for item in data:
    total = total + item * 2
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_conditional_in_loop() {
        let code = r#"
count = 0
for i in range(10):
    if i % 2 == 0:
        count = count + 1
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_list_building() {
        let code = r#"
result = []
for i in range(5):
    result = result + [i * i]
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_dict_building() {
        let code = r#"
data = {}
data["a"] = 1
data["b"] = 2
data["c"] = 3
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_mixed_types() {
        let code = r#"
x = 1
y = 2.5
z = x + y
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_deeply_nested_structures() {
        let code = r#"
data = [[1, 2], [3, 4], [5, 6]]
x = data[0][0]
y = data[1][1]
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_method_chaining_simulation() {
        let code = r#"
class Builder:
    def __init__(self):
        self.value = 0
    def add(self, x):
        self.value = self.value + x
        return self
b = Builder()
b.add(1)
b.add(2)
b.add(3)
"#;
        assert!(run(code).is_ok());
    }

    #[test]
    fn test_print_statement() {
        // print returns None but shouldn't error
        assert!(run("print(42)").is_ok());
        assert!(run(r#"print("hello")"#).is_ok());
        assert!(run("print(1, 2, 3)").is_ok());
    }
}

// ============================================================================
// Tuple Iteration Tests
// ============================================================================

#[test]
fn test_tuple_iteration() {
    let code = r#"
t = (1, 2, 3)
total = 0
for x in t:
    total = total + x
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_tuple_iteration_strings() {
    let code = r#"
t = ("a", "b", "c")
result = []
for s in t:
    result = result + [s]
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_tuple_iteration_single() {
    let code = r#"
t = (42,)
total = 0
for x in t:
    total = total + x
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_tuple_iteration_nested() {
    let code = r#"
pairs = ((1, 2), (3, 4))
total = 0
for p in pairs:
    total = total + 1
"#;
    assert!(run(code).is_ok());
}

// ============================================================================
// Dict Iteration Tests
// ============================================================================

#[test]
fn test_dict_iteration_keys() {
    let code = "d = {\"a\": 1, \"b\": 2}\nresult = []\nfor k in d:\n    result = result + [k]";
    assert!(run(code).is_ok());
}

#[test]
fn test_dict_iteration_empty() {
    let code = "d = {}\ncount = 0\nfor k in d:\n    count = count + 1";
    assert!(run(code).is_ok());
}

#[test]
fn test_dict_iteration_access_values() {
    let code = "d = {\"x\": 10, \"y\": 20}\ntotal = 0\nfor k in d:\n    total = total + d[k]";
    assert!(run(code).is_ok());
}

// ============================================================================
// Map/Filter Builtin Tests
// ============================================================================

#[test]
fn test_map_with_native_function() {
    let code = r#"
result = map(str, [1, 2, 3])
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_map_with_int_conversion() {
    let code = r#"
result = map(int, [1.5, 2.7, 3.1])
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_map_with_len() {
    let code = r#"
result = map(len, ["hi", "hello", "hey"])
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_filter_none_truthy() {
    let code = r#"
result = filter(None, [0, 1, 2, 0, 3])
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_filter_none_strings() {
    let code = r#"
result = filter(None, ["", "a", "", "b"])
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_filter_with_native_bool() {
    let code = r#"
result = filter(bool, [0, 1, False, True, "", "x"])
"#;
    assert!(run(code).is_ok());
}

// ============================================================================
// isinstance Tests
// ============================================================================

#[test]
fn test_isinstance_string_type() {
    let code = r#"
x = isinstance(42, "int")
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_isinstance_class() {
    let code = r#"
class Animal:
    def __init__(self):
        self.name = "animal"
a = Animal()
result = isinstance(a, Animal)
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_isinstance_wrong_class() {
    let code = r#"
class Cat:
    def __init__(self):
        pass
class Dog:
    def __init__(self):
        pass
c = Cat()
result = isinstance(c, Dog)
"#;
    assert!(run(code).is_ok());
}

// ============================================================================
// Dict Subscript Operation Tests
// ============================================================================

#[test]
fn test_dict_subscript_read() {
    let code = "d = {\"key\": 42}\nval = d[\"key\"]";
    assert!(run(code).is_ok());
}

#[test]
fn test_dict_subscript_write() {
    let code = "d = {}\nd[\"new\"] = 99";
    assert!(run(code).is_ok());
}

#[test]
fn test_dict_subscript_overwrite() {
    let code = "d = {\"a\": 1}\nd[\"a\"] = 2";
    assert!(run(code).is_ok());
}

#[test]
fn test_dict_subscript_multiple() {
    let code = "d = {}\nd[\"x\"] = 1\nd[\"y\"] = 2\nd[\"z\"] = 3\nval = d[\"y\"]";
    assert!(run(code).is_ok());
}

// ============================================================================
// String Iteration Tests
// ============================================================================

#[test]
fn test_string_iteration() {
    let code = r#"
s = "abc"
chars = []
for c in s:
    chars = chars + [c]
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_string_iteration_empty() {
    let code = r#"
s = ""
count = 0
for c in s:
    count = count + 1
"#;
    assert!(run(code).is_ok());
}
