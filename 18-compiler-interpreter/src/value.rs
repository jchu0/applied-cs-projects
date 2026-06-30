//! Runtime values.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use crate::compiler::CodeObject;

/// Runtime value.
#[derive(Clone)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(Rc<String>),
    List(Rc<RefCell<Vec<Value>>>),
    Dict(Rc<RefCell<HashMap<String, Value>>>),
    Tuple(Rc<Vec<Value>>),
    Function(Rc<Function>),
    NativeFunction(Rc<NativeFunction>),
    Class(Rc<Class>),
    Instance(Rc<RefCell<Instance>>),
    BoundMethod(Rc<BoundMethod>),
    Range(Rc<RangeValue>),
    Iterator(Rc<RefCell<Box<dyn ValueIterator>>>),
    Generator(Rc<RefCell<Generator>>),
    Coroutine(Rc<RefCell<Coroutine>>),
    // Descriptor types for decorators
    Property(Rc<PropertyDescriptor>),
    StaticMethod(Rc<StaticMethodDescriptor>),
    ClassMethod(Rc<ClassMethodDescriptor>),
    // Bound list method
    BoundListMethod(Rc<RefCell<Vec<Value>>>, String),
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::None => write!(f, "None"),
            Value::Bool(b) => write!(f, "{}", if *b { "True" } else { "False" }),
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::String(s) => write!(f, "'{}'", s),
            Value::List(l) => write!(f, "{:?}", l.borrow()),
            Value::Dict(d) => write!(f, "{:?}", d.borrow()),
            Value::Tuple(t) => write!(f, "{:?}", t),
            Value::Function(func) => write!(f, "<function {}>", func.name),
            Value::NativeFunction(func) => write!(f, "<built-in function {}>", func.name),
            Value::Class(c) => write!(f, "<class '{}'>", c.name),
            Value::Instance(i) => write!(f, "<{} instance>", i.borrow().class.name),
            Value::BoundMethod(_) => write!(f, "<bound method>"),
            Value::Range(r) => write!(f, "range({}, {}, {})", r.start, r.stop, r.step),
            Value::Iterator(_) => write!(f, "<iterator>"),
            Value::Generator(g) => write!(f, "<generator object {}>", g.borrow().name),
            Value::Coroutine(c) => write!(f, "<coroutine object {}>", c.borrow().name),
            Value::Property(_) => write!(f, "<property>"),
            Value::StaticMethod(_) => write!(f, "<staticmethod>"),
            Value::ClassMethod(_) => write!(f, "<classmethod>"),
            Value::BoundListMethod(_, name) => write!(f, "<built-in method {}>", name),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::None, Value::None) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Int(a), Value::Float(b)) => (*a as f64) == *b,
            (Value::Float(a), Value::Int(b)) => *a == (*b as f64),
            (Value::String(a), Value::String(b)) => a == b,
            (Value::List(a), Value::List(b)) => *a.borrow() == *b.borrow(),
            (Value::Tuple(a), Value::Tuple(b)) => a == b,
            _ => false,
        }
    }
}

impl Value {
    /// Check if value is truthy.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::None => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::String(s) => !s.is_empty(),
            Value::List(l) => !l.borrow().is_empty(),
            Value::Dict(d) => !d.borrow().is_empty(),
            Value::Tuple(t) => !t.is_empty(),
            _ => true,
        }
    }

    /// Get type name.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::None => "NoneType",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) => "str",
            Value::List(_) => "list",
            Value::Dict(_) => "dict",
            Value::Tuple(_) => "tuple",
            Value::Function(_) | Value::NativeFunction(_) => "function",
            Value::Class(_) => "type",
            Value::Instance(_) => "object",
            Value::BoundMethod(_) => "method",
            Value::Range(_) => "range",
            Value::Iterator(_) => "iterator",
            Value::Generator(_) => "generator",
            Value::Coroutine(_) => "coroutine",
            Value::Property(_) => "property",
            Value::StaticMethod(_) => "staticmethod",
            Value::ClassMethod(_) => "classmethod",
            Value::BoundListMethod(_, _) => "builtin_method",
        }
    }

    /// Get string representation (for exceptions and debugging).
    pub fn repr(&self) -> String {
        format!("{:?}", self)
    }
}

/// User-defined function.
#[derive(Debug)]
pub struct Function {
    pub name: String,
    pub code: Rc<CodeObject>,
    pub defaults: Vec<Value>,
    pub closure: Option<Rc<RefCell<Vec<Value>>>>,
}

/// Native (built-in) function.
pub struct NativeFunction {
    pub name: String,
    pub arity: i32, // -1 for variadic
    pub func: fn(&[Value]) -> crate::Result<Value>,
}

impl fmt::Debug for NativeFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<native function {}>", self.name)
    }
}

/// Class definition.
#[derive(Debug)]
pub struct Class {
    pub name: String,
    pub bases: Vec<Rc<Class>>,
    pub methods: HashMap<String, Value>,
}

/// Class instance.
#[derive(Debug)]
pub struct Instance {
    pub class: Rc<Class>,
    pub fields: HashMap<String, Value>,
}

/// Bound method.
#[derive(Debug)]
pub struct BoundMethod {
    pub receiver: Value,
    pub method: Value,
}

/// Range value.
#[derive(Debug)]
pub struct RangeValue {
    pub start: i64,
    pub stop: i64,
    pub step: i64,
}

/// Iterator trait for values.
pub trait ValueIterator: fmt::Debug {
    fn next_value(&mut self) -> Option<Value>;
}

/// Range iterator.
#[derive(Debug)]
pub struct RangeIterator {
    current: i64,
    stop: i64,
    step: i64,
}

impl RangeIterator {
    pub fn new(start: i64, stop: i64, step: i64) -> Self {
        Self {
            current: start,
            stop,
            step,
        }
    }
}

impl ValueIterator for RangeIterator {
    fn next_value(&mut self) -> Option<Value> {
        if (self.step > 0 && self.current < self.stop)
            || (self.step < 0 && self.current > self.stop)
        {
            let value = self.current;
            self.current += self.step;
            Some(Value::Int(value))
        } else {
            None
        }
    }
}

/// List iterator.
#[derive(Debug)]
pub struct ListIterator {
    list: Rc<RefCell<Vec<Value>>>,
    index: usize,
}

impl ListIterator {
    pub fn new(list: Rc<RefCell<Vec<Value>>>) -> Self {
        Self { list, index: 0 }
    }
}

impl ValueIterator for ListIterator {
    fn next_value(&mut self) -> Option<Value> {
        let list = self.list.borrow();
        if self.index < list.len() {
            let value = list[self.index].clone();
            self.index += 1;
            Some(value)
        } else {
            None
        }
    }
}

/// String iterator.
#[derive(Debug)]
pub struct StringIterator {
    chars: Vec<char>,
    index: usize,
}

impl StringIterator {
    pub fn new(s: &str) -> Self {
        Self {
            chars: s.chars().collect(),
            index: 0,
        }
    }
}

impl ValueIterator for StringIterator {
    fn next_value(&mut self) -> Option<Value> {
        if self.index < self.chars.len() {
            let c = self.chars[self.index];
            self.index += 1;
            Some(Value::String(Rc::new(c.to_string())))
        } else {
            None
        }
    }
}

/// Tuple iterator.
#[derive(Debug)]
pub struct TupleIterator {
    items: Rc<Vec<Value>>,
    index: usize,
}

impl TupleIterator {
    pub fn new(items: Rc<Vec<Value>>) -> Self {
        Self { items, index: 0 }
    }
}

impl ValueIterator for TupleIterator {
    fn next_value(&mut self) -> Option<Value> {
        if self.index < self.items.len() {
            let value = self.items[self.index].clone();
            self.index += 1;
            Some(value)
        } else {
            None
        }
    }
}

/// Dict key iterator (iterates over keys).
#[derive(Debug)]
pub struct DictIterator {
    keys: Vec<String>,
    index: usize,
}

impl DictIterator {
    pub fn new(dict: &Rc<RefCell<HashMap<String, Value>>>) -> Self {
        let keys: Vec<String> = dict.borrow().keys().cloned().collect();
        Self { keys, index: 0 }
    }
}

impl ValueIterator for DictIterator {
    fn next_value(&mut self) -> Option<Value> {
        if self.index < self.keys.len() {
            let key = self.keys[self.index].clone();
            self.index += 1;
            Some(Value::String(Rc::new(key)))
        } else {
            None
        }
    }
}

/// Generator state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorState {
    Created,
    Running,
    Suspended,
    Completed,
}

/// Generator object for yield-based iteration.
#[derive(Debug)]
pub struct Generator {
    pub name: String,
    pub code: Rc<CodeObject>,
    pub state: GeneratorState,
    pub ip: usize,                           // Instruction pointer
    pub stack: Vec<Value>,                   // Local stack
    pub locals: HashMap<String, Value>,      // Local variables
    pub send_value: Option<Value>,           // Value sent into generator
}

impl Generator {
    pub fn new(name: String, code: Rc<CodeObject>) -> Self {
        Self {
            name,
            code,
            state: GeneratorState::Created,
            ip: 0,
            stack: Vec::new(),
            locals: HashMap::new(),
            send_value: None,
        }
    }

    /// Check if the generator is exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.state == GeneratorState::Completed
    }
}

/// Generator iterator wrapper.
#[derive(Debug)]
pub struct GeneratorIterator {
    pub generator: Rc<RefCell<Generator>>,
}

impl GeneratorIterator {
    pub fn new(generator: Rc<RefCell<Generator>>) -> Self {
        Self { generator }
    }
}

impl ValueIterator for GeneratorIterator {
    fn next_value(&mut self) -> Option<Value> {
        let gen = self.generator.borrow();
        if gen.is_exhausted() {
            return None;
        }
        drop(gen);
        // Actual iteration is handled by VM
        None
    }
}

/// Coroutine state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoroutineState {
    Created,
    Running,
    Suspended,
    Completed,
}

/// Coroutine object for async/await.
#[derive(Debug)]
pub struct Coroutine {
    pub name: String,
    pub code: Rc<CodeObject>,
    pub state: CoroutineState,
    pub ip: usize,                           // Instruction pointer
    pub stack: Vec<Value>,                   // Local stack
    pub locals: HashMap<String, Value>,      // Local variables
    pub send_value: Option<Value>,           // Value sent into coroutine
    pub awaiting: Option<Value>,             // Currently awaiting value
}

impl Coroutine {
    pub fn new(name: String, code: Rc<CodeObject>) -> Self {
        Self {
            name,
            code,
            state: CoroutineState::Created,
            ip: 0,
            stack: Vec::new(),
            locals: HashMap::new(),
            send_value: None,
            awaiting: None,
        }
    }

    /// Check if the coroutine is done.
    pub fn is_done(&self) -> bool {
        self.state == CoroutineState::Completed
    }
}

/// Property descriptor for @property decorator.
/// Provides getter, setter, and deleter for attribute access.
#[derive(Debug, Clone)]
pub struct PropertyDescriptor {
    pub fget: Option<Value>,   // Getter function
    pub fset: Option<Value>,   // Setter function
    pub fdel: Option<Value>,   // Deleter function
    pub doc: Option<String>,   // Docstring
}

impl PropertyDescriptor {
    pub fn new(fget: Option<Value>) -> Self {
        Self {
            fget,
            fset: None,
            fdel: None,
            doc: None,
        }
    }

    pub fn with_setter(mut self, fset: Value) -> Self {
        self.fset = Some(fset);
        self
    }

    pub fn with_deleter(mut self, fdel: Value) -> Self {
        self.fdel = Some(fdel);
        self
    }
}

/// StaticMethod descriptor for @staticmethod decorator.
/// Wraps a function to not receive self/cls as first argument.
#[derive(Debug, Clone)]
pub struct StaticMethodDescriptor {
    pub func: Value,
}

impl StaticMethodDescriptor {
    pub fn new(func: Value) -> Self {
        Self { func }
    }
}

/// ClassMethod descriptor for @classmethod decorator.
/// Wraps a function to receive class as first argument.
#[derive(Debug, Clone)]
pub struct ClassMethodDescriptor {
    pub func: Value,
}

impl ClassMethodDescriptor {
    pub fn new(func: Value) -> Self {
        Self { func }
    }
}
