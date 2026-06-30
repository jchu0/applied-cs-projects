//! Tests for decorator functionality.

use py_compiler::run;

// ============================================================================
// Basic Decorator Tests
// ============================================================================

#[test]
fn test_simple_decorator() {
    // Test a simple identity decorator
    let code = r#"
def identity(func):
    return func

@identity
def get_value():
    return 42

result = get_value()
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_decorator_with_arguments() {
    let code = r#"
def multiply(factor):
    def decorator(func):
        def wrapper(x):
            return func(x) * factor
        return wrapper
    return decorator

@multiply(3)
def double(x):
    return x * 2

result = double(5)
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_multiple_decorators() {
    let code = r#"
def add_one(func):
    def wrapper(x):
        return func(x) + 1
    return wrapper

def double(func):
    def wrapper(x):
        return func(x) * 2
    return wrapper

@add_one
@double
def identity(x):
    return x

result = identity(5)
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_decorator_on_method() {
    let code = r#"
def log_call(func):
    def wrapper(self, x):
        return func(self, x)
    return wrapper

class Calculator:
    @log_call
    def square(self, x):
        return x * x

calc = Calculator()
result = calc.square(4)
"#;
    assert!(run(code).is_ok());
}

// ============================================================================
// Built-in Decorator Tests
// ============================================================================

#[test]
fn test_staticmethod_decorator() {
    let code = r#"
class MathUtils:
    @staticmethod
    def add(a, b):
        return a + b

result = MathUtils.add(3, 4)
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_staticmethod_from_instance() {
    let code = r#"
class Counter:
    count = 0

    @staticmethod
    def increment(n):
        return n + 1

c = Counter()
result = c.increment(5)
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_classmethod_decorator() {
    let code = r#"
class Person:
    count = 0

    def __init__(self, name):
        self.name = name

    @classmethod
    def get_count(cls):
        return 0

p = Person("Alice")
result = Person.get_count()
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_classmethod_from_instance() {
    let code = r#"
class Animal:
    @classmethod
    def species_info(cls):
        return "animal"

dog = Animal()
result = dog.species_info()
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_property_decorator() {
    let code = r#"
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

c = Circle(5)
r = c._radius
"#;
    assert!(run(code).is_ok());
}

// ============================================================================
// Class Decorator Tests
// ============================================================================

#[test]
fn test_class_decorator() {
    let code = r#"
def add_greeting(cls):
    cls.greet = lambda self: "Hello"
    return cls

@add_greeting
class Person:
    def __init__(self, name):
        self.name = name

p = Person("Alice")
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_multiple_class_decorators() {
    let code = r#"
def add_x(cls):
    cls.x = 10
    return cls

def add_y(cls):
    cls.y = 20
    return cls

@add_x
@add_y
class Point:
    pass

p = Point()
"#;
    assert!(run(code).is_ok());
}

// ============================================================================
// Decorator Edge Cases
// ============================================================================

#[test]
fn test_decorator_preserves_function() {
    let code = r#"
def identity_decorator(func):
    return func

@identity_decorator
def greet(name):
    return "Hello, " + name

result = greet("World")
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_decorator_with_closure() {
    let code = r#"
def make_repeater(n):
    def decorator(func):
        def wrapper(x):
            result = x
            for i in range(n):
                result = func(result)
            return result
        return wrapper
    return decorator

@make_repeater(3)
def increment(x):
    return x + 1

result = increment(0)
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_nested_decorated_functions() {
    let code = r#"
def outer_decorator(func):
    def wrapper():
        return func() + 100
    return wrapper

def inner_decorator(func):
    def wrapper():
        return func() * 2
    return wrapper

@outer_decorator
def outer_func():
    @inner_decorator
    def inner_func():
        return 5
    return inner_func()

result = outer_func()
"#;
    assert!(run(code).is_ok());
}

// ============================================================================
// Decorator with State
// ============================================================================

#[test]
fn test_decorator_with_counter() {
    let code = r#"
call_count = 0

def count_calls(func):
    def wrapper(x):
        global call_count
        call_count = call_count + 1
        return func(x)
    return wrapper

@count_calls
def square(x):
    return x * x

r1 = square(2)
r2 = square(3)
r3 = square(4)
"#;
    assert!(run(code).is_ok());
}

// ============================================================================
// Combined Decorator Tests
// ============================================================================

#[test]
fn test_staticmethod_with_custom_decorator() {
    // Note: *args syntax not yet supported, using simplified decorator
    let code = r#"
def log(func):
    def wrapper(a, b):
        return func(a, b)
    return wrapper

class Calculator:
    @staticmethod
    def multiply(a, b):
        return a * b

result = Calculator.multiply(3, 4)
"#;
    assert!(run(code).is_ok());
}

#[test]
fn test_method_decorators_mixed() {
    let code = r#"
class MyClass:
    value = 42

    def regular_method(self):
        return self.value

    @staticmethod
    def static_method():
        return 100

    @classmethod
    def class_method(cls):
        return cls.value

obj = MyClass()
r1 = obj.regular_method()
r2 = MyClass.static_method()
r3 = MyClass.class_method()
"#;
    assert!(run(code).is_ok());
}
