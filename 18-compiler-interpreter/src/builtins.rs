//! Built-in functions.

use crate::value::{
    ClassMethodDescriptor, NativeFunction, PropertyDescriptor, RangeValue,
    StaticMethodDescriptor, Value,
};
use crate::{Error, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Register all built-in functions.
pub fn register_builtins(globals: &mut HashMap<String, Value>) {
    macro_rules! builtin {
        ($name:expr, $arity:expr, $func:expr) => {
            globals.insert(
                $name.to_string(),
                Value::NativeFunction(Rc::new(NativeFunction {
                    name: $name.to_string(),
                    arity: $arity,
                    func: $func,
                })),
            );
        };
    }

    builtin!("print", -1, builtin_print);
    builtin!("len", 1, builtin_len);
    builtin!("range", -1, builtin_range);
    builtin!("int", 1, builtin_int);
    builtin!("float", 1, builtin_float);
    builtin!("str", 1, builtin_str);
    builtin!("bool", 1, builtin_bool);
    builtin!("list", 1, builtin_list);
    builtin!("dict", 0, builtin_dict);
    builtin!("tuple", 1, builtin_tuple);
    builtin!("type", 1, builtin_type);
    builtin!("abs", 1, builtin_abs);
    builtin!("min", -1, builtin_min);
    builtin!("max", -1, builtin_max);
    builtin!("sum", -1, builtin_sum);
    builtin!("round", -1, builtin_round);
    builtin!("sorted", 1, builtin_sorted);
    builtin!("reversed", 1, builtin_reversed);
    builtin!("enumerate", 1, builtin_enumerate);
    builtin!("zip", -1, builtin_zip);
    builtin!("map", -1, builtin_map);
    builtin!("filter", -1, builtin_filter);
    builtin!("input", -1, builtin_input);
    builtin!("ord", 1, builtin_ord);
    builtin!("chr", 1, builtin_chr);
    builtin!("isinstance", 2, builtin_isinstance);
    builtin!("hasattr", 2, builtin_hasattr);
    builtin!("getattr", -1, builtin_getattr);
    builtin!("setattr", 3, builtin_setattr);

    // Decorator built-ins
    builtin!("property", -1, builtin_property);
    builtin!("staticmethod", 1, builtin_staticmethod);
    builtin!("classmethod", 1, builtin_classmethod);
}

fn builtin_print(args: &[Value]) -> Result<Value> {
    let output: Vec<String> = args.iter().map(|v| format_value(v)).collect();
    println!("{}", output.join(" "));
    Ok(Value::None)
}

fn builtin_len(args: &[Value]) -> Result<Value> {
    match &args[0] {
        Value::String(s) => Ok(Value::Int(s.len() as i64)),
        Value::List(l) => Ok(Value::Int(l.borrow().len() as i64)),
        Value::Dict(d) => Ok(Value::Int(d.borrow().len() as i64)),
        Value::Tuple(t) => Ok(Value::Int(t.len() as i64)),
        _ => Err(Error::Type(format!(
            "object of type '{}' has no len()",
            args[0].type_name()
        ))),
    }
}

fn builtin_range(args: &[Value]) -> Result<Value> {
    let (start, stop, step) = match args.len() {
        1 => {
            let stop = match &args[0] {
                Value::Int(n) => *n,
                _ => return Err(Error::Type("range() argument must be int".into())),
            };
            (0, stop, 1)
        }
        2 => {
            let start = match &args[0] {
                Value::Int(n) => *n,
                _ => return Err(Error::Type("range() argument must be int".into())),
            };
            let stop = match &args[1] {
                Value::Int(n) => *n,
                _ => return Err(Error::Type("range() argument must be int".into())),
            };
            (start, stop, 1)
        }
        3 => {
            let start = match &args[0] {
                Value::Int(n) => *n,
                _ => return Err(Error::Type("range() argument must be int".into())),
            };
            let stop = match &args[1] {
                Value::Int(n) => *n,
                _ => return Err(Error::Type("range() argument must be int".into())),
            };
            let step = match &args[2] {
                Value::Int(n) => *n,
                _ => return Err(Error::Type("range() argument must be int".into())),
            };
            if step == 0 {
                return Err(Error::Value("range() step argument must not be zero".into()));
            }
            (start, stop, step)
        }
        _ => {
            return Err(Error::Type(
                "range expected at most 3 arguments".into(),
            ));
        }
    };

    Ok(Value::Range(Rc::new(RangeValue { start, stop, step })))
}

fn builtin_int(args: &[Value]) -> Result<Value> {
    match &args[0] {
        Value::Int(n) => Ok(Value::Int(*n)),
        Value::Float(f) => Ok(Value::Int(*f as i64)),
        Value::String(s) => s
            .trim()
            .parse::<i64>()
            .map(Value::Int)
            .map_err(|_| Error::Value(format!("invalid literal for int(): '{}'", s))),
        Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
        _ => Err(Error::Type(format!(
            "int() argument must be a string or a number, not '{}'",
            args[0].type_name()
        ))),
    }
}

fn builtin_float(args: &[Value]) -> Result<Value> {
    match &args[0] {
        Value::Float(f) => Ok(Value::Float(*f)),
        Value::Int(n) => Ok(Value::Float(*n as f64)),
        Value::String(s) => s
            .trim()
            .parse::<f64>()
            .map(Value::Float)
            .map_err(|_| Error::Value(format!("could not convert string to float: '{}'", s))),
        _ => Err(Error::Type(format!(
            "float() argument must be a string or a number, not '{}'",
            args[0].type_name()
        ))),
    }
}

fn builtin_str(args: &[Value]) -> Result<Value> {
    Ok(Value::String(Rc::new(format_value(&args[0]))))
}

fn builtin_bool(args: &[Value]) -> Result<Value> {
    Ok(Value::Bool(args[0].is_truthy()))
}

fn builtin_list(args: &[Value]) -> Result<Value> {
    match &args[0] {
        Value::List(l) => Ok(Value::List(Rc::new(RefCell::new(l.borrow().clone())))),
        Value::Tuple(t) => Ok(Value::List(Rc::new(RefCell::new((**t).clone())))),
        Value::String(s) => {
            let chars: Vec<Value> = s
                .chars()
                .map(|c| Value::String(Rc::new(c.to_string())))
                .collect();
            Ok(Value::List(Rc::new(RefCell::new(chars))))
        }
        Value::Range(r) => {
            let mut values = Vec::new();
            let mut i = r.start;
            while (r.step > 0 && i < r.stop) || (r.step < 0 && i > r.stop) {
                values.push(Value::Int(i));
                i += r.step;
            }
            Ok(Value::List(Rc::new(RefCell::new(values))))
        }
        _ => Err(Error::Type(format!(
            "'{}' object is not iterable",
            args[0].type_name()
        ))),
    }
}

fn builtin_dict(args: &[Value]) -> Result<Value> {
    Ok(Value::Dict(Rc::new(RefCell::new(HashMap::new()))))
}

fn builtin_tuple(args: &[Value]) -> Result<Value> {
    match &args[0] {
        Value::Tuple(t) => Ok(Value::Tuple(t.clone())),
        Value::List(l) => Ok(Value::Tuple(Rc::new(l.borrow().clone()))),
        _ => Err(Error::Type("tuple() argument must be iterable".into())),
    }
}

fn builtin_type(args: &[Value]) -> Result<Value> {
    Ok(Value::String(Rc::new(args[0].type_name().to_string())))
}

fn builtin_abs(args: &[Value]) -> Result<Value> {
    match &args[0] {
        Value::Int(n) => Ok(Value::Int(n.abs())),
        Value::Float(f) => Ok(Value::Float(f.abs())),
        _ => Err(Error::Type("abs() requires numeric argument".into())),
    }
}

fn builtin_min(args: &[Value]) -> Result<Value> {
    if args.is_empty() {
        return Err(Error::Type("min expected 1 arguments, got 0".into()));
    }

    // If single iterable argument
    if args.len() == 1 {
        match &args[0] {
            Value::List(l) => {
                let list = l.borrow();
                if list.is_empty() {
                    return Err(Error::Value("min() arg is an empty sequence".into()));
                }
                return find_min(&list);
            }
            _ => {}
        }
    }

    find_min(args)
}

fn builtin_max(args: &[Value]) -> Result<Value> {
    if args.is_empty() {
        return Err(Error::Type("max expected 1 arguments, got 0".into()));
    }

    if args.len() == 1 {
        match &args[0] {
            Value::List(l) => {
                let list = l.borrow();
                if list.is_empty() {
                    return Err(Error::Value("max() arg is an empty sequence".into()));
                }
                return find_max(&list);
            }
            _ => {}
        }
    }

    find_max(args)
}

fn builtin_sum(args: &[Value]) -> Result<Value> {
    let iterable = match &args[0] {
        Value::List(l) => l.borrow().clone(),
        Value::Tuple(t) => (**t).clone(),
        _ => return Err(Error::Type("sum() argument must be iterable".into())),
    };

    let mut total = 0i64;
    let mut is_float = false;
    let mut float_total = 0.0f64;

    for item in iterable {
        match item {
            Value::Int(n) => {
                if is_float {
                    float_total += n as f64;
                } else {
                    total += n;
                }
            }
            Value::Float(f) => {
                if !is_float {
                    is_float = true;
                    float_total = total as f64;
                }
                float_total += f;
            }
            _ => return Err(Error::Type("unsupported operand type for sum".into())),
        }
    }

    if is_float {
        Ok(Value::Float(float_total))
    } else {
        Ok(Value::Int(total))
    }
}

fn builtin_round(args: &[Value]) -> Result<Value> {
    let value = match &args[0] {
        Value::Int(n) => return Ok(Value::Int(*n)),
        Value::Float(f) => *f,
        _ => return Err(Error::Type("round() requires numeric argument".into())),
    };

    let digits = if args.len() > 1 {
        match &args[1] {
            Value::Int(n) => *n as i32,
            _ => return Err(Error::Type("round() digits must be int".into())),
        }
    } else {
        0
    };

    if digits == 0 {
        Ok(Value::Int(value.round() as i64))
    } else {
        let multiplier = 10f64.powi(digits);
        Ok(Value::Float((value * multiplier).round() / multiplier))
    }
}

fn builtin_sorted(args: &[Value]) -> Result<Value> {
    let mut items = match &args[0] {
        Value::List(l) => l.borrow().clone(),
        Value::Tuple(t) => (**t).clone(),
        _ => return Err(Error::Type("sorted() argument must be iterable".into())),
    };

    // Simple sort for numbers and strings
    items.sort_by(|a, b| {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => x.cmp(y),
            (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
            (Value::String(x), Value::String(y)) => x.cmp(y),
            _ => std::cmp::Ordering::Equal,
        }
    });

    Ok(Value::List(Rc::new(RefCell::new(items))))
}

fn builtin_reversed(args: &[Value]) -> Result<Value> {
    let mut items = match &args[0] {
        Value::List(l) => l.borrow().clone(),
        Value::Tuple(t) => (**t).clone(),
        Value::String(s) => s
            .chars()
            .map(|c| Value::String(Rc::new(c.to_string())))
            .collect(),
        _ => return Err(Error::Type("reversed() argument must be sequence".into())),
    };

    items.reverse();
    Ok(Value::List(Rc::new(RefCell::new(items))))
}

fn builtin_enumerate(args: &[Value]) -> Result<Value> {
    let items = match &args[0] {
        Value::List(l) => l.borrow().clone(),
        Value::Tuple(t) => (**t).clone(),
        _ => return Err(Error::Type("enumerate() argument must be iterable".into())),
    };

    let result: Vec<Value> = items
        .into_iter()
        .enumerate()
        .map(|(i, v)| Value::Tuple(Rc::new(vec![Value::Int(i as i64), v])))
        .collect();

    Ok(Value::List(Rc::new(RefCell::new(result))))
}

fn builtin_zip(args: &[Value]) -> Result<Value> {
    if args.is_empty() {
        return Ok(Value::List(Rc::new(RefCell::new(Vec::new()))));
    }

    let iterables: Vec<Vec<Value>> = args
        .iter()
        .map(|arg| match arg {
            Value::List(l) => Ok(l.borrow().clone()),
            Value::Tuple(t) => Ok((**t).clone()),
            _ => Err(Error::Type("zip() argument must be iterable".into())),
        })
        .collect::<Result<Vec<_>>>()?;

    let min_len = iterables.iter().map(|v| v.len()).min().unwrap_or(0);
    let mut result = Vec::new();

    for i in 0..min_len {
        let tuple: Vec<Value> = iterables.iter().map(|v| v[i].clone()).collect();
        result.push(Value::Tuple(Rc::new(tuple)));
    }

    Ok(Value::List(Rc::new(RefCell::new(result))))
}

fn builtin_map(args: &[Value]) -> Result<Value> {
    if args.len() < 2 {
        return Err(Error::Type("map() requires at least two arguments".into()));
    }
    let func = &args[0];
    let items = iterable_to_vec(&args[1])?;

    let mut result = Vec::new();
    match func {
        Value::NativeFunction(nf) => {
            for item in &items {
                let val = (nf.func)(&[item.clone()])?;
                result.push(val);
            }
        }
        _ => {
            return Err(Error::Type(
                "map() with non-builtin functions requires list comprehension syntax".into(),
            ));
        }
    }
    Ok(Value::List(Rc::new(RefCell::new(result))))
}

fn builtin_filter(args: &[Value]) -> Result<Value> {
    if args.len() < 2 {
        return Err(Error::Type("filter() requires two arguments".into()));
    }
    let func = &args[0];
    let items = iterable_to_vec(&args[1])?;

    let mut result = Vec::new();
    match func {
        Value::None => {
            // filter(None, iterable) keeps truthy values
            for item in items {
                if item.is_truthy() {
                    result.push(item);
                }
            }
        }
        Value::NativeFunction(nf) => {
            for item in &items {
                let val = (nf.func)(&[item.clone()])?;
                if val.is_truthy() {
                    result.push(item.clone());
                }
            }
        }
        _ => {
            return Err(Error::Type(
                "filter() with non-builtin functions requires list comprehension syntax".into(),
            ));
        }
    }
    Ok(Value::List(Rc::new(RefCell::new(result))))
}

/// Convert an iterable Value to a Vec<Value>.
fn iterable_to_vec(value: &Value) -> Result<Vec<Value>> {
    match value {
        Value::List(l) => Ok(l.borrow().clone()),
        Value::Tuple(t) => Ok((**t).clone()),
        Value::String(s) => Ok(s
            .chars()
            .map(|c| Value::String(Rc::new(c.to_string())))
            .collect()),
        Value::Range(r) => {
            let mut items = Vec::new();
            let mut current = r.start;
            while (r.step > 0 && current < r.stop) || (r.step < 0 && current > r.stop) {
                items.push(Value::Int(current));
                current += r.step;
            }
            Ok(items)
        }
        _ => Err(Error::Type(format!(
            "'{}' object is not iterable",
            value.type_name()
        ))),
    }
}

fn builtin_input(args: &[Value]) -> Result<Value> {
    if !args.is_empty() {
        print!("{}", format_value(&args[0]));
    }

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .map_err(|e| Error::Io(e))?;

    Ok(Value::String(Rc::new(input.trim_end().to_string())))
}

fn builtin_ord(args: &[Value]) -> Result<Value> {
    match &args[0] {
        Value::String(s) => {
            let chars: Vec<char> = s.chars().collect();
            if chars.len() != 1 {
                return Err(Error::Type(format!(
                    "ord() expected a character, but string of length {} found",
                    chars.len()
                )));
            }
            Ok(Value::Int(chars[0] as i64))
        }
        _ => Err(Error::Type("ord() expected string of length 1".into())),
    }
}

fn builtin_chr(args: &[Value]) -> Result<Value> {
    match &args[0] {
        Value::Int(n) => {
            if *n < 0 || *n > 0x10ffff {
                return Err(Error::Value(format!("chr() arg not in range(0x110000)")));
            }
            let c = char::from_u32(*n as u32).ok_or_else(|| {
                Error::Value(format!("chr() arg not a valid character"))
            })?;
            Ok(Value::String(Rc::new(c.to_string())))
        }
        _ => Err(Error::Type("chr() requires int argument".into())),
    }
}

fn builtin_isinstance(args: &[Value]) -> Result<Value> {
    if args.len() < 2 {
        return Err(Error::Type("isinstance() requires two arguments".into()));
    }

    let obj = &args[0];
    let type_arg = &args[1];

    match type_arg {
        // isinstance(x, ClassName) - check against a class
        Value::Class(cls) => {
            if let Value::Instance(inst) = obj {
                let inst = inst.borrow();
                // Check direct class match
                if inst.class.name == cls.name {
                    return Ok(Value::Bool(true));
                }
                // Check base classes (MRO)
                for base in &inst.class.bases {
                    if base.name == cls.name {
                        return Ok(Value::Bool(true));
                    }
                }
                Ok(Value::Bool(false))
            } else {
                Ok(Value::Bool(false))
            }
        }
        // isinstance(x, "typename") - check against type name string
        Value::String(s) => Ok(Value::Bool(obj.type_name() == s.as_str())),
        // isinstance(x, (type1, type2, ...)) - check against tuple of types
        Value::Tuple(types) => {
            for t in types.iter() {
                let result = builtin_isinstance(&[obj.clone(), t.clone()])?;
                if let Value::Bool(true) = result {
                    return Ok(Value::Bool(true));
                }
            }
            Ok(Value::Bool(false))
        }
        _ => Ok(Value::Bool(false)),
    }
}

fn builtin_hasattr(args: &[Value]) -> Result<Value> {
    match &args[0] {
        Value::Instance(inst) => {
            if let Value::String(name) = &args[1] {
                let inst = inst.borrow();
                let has = inst.fields.contains_key(&**name)
                    || inst.class.methods.contains_key(&**name);
                Ok(Value::Bool(has))
            } else {
                Err(Error::Type("hasattr() attribute name must be string".into()))
            }
        }
        _ => Ok(Value::Bool(false)),
    }
}

fn builtin_getattr(args: &[Value]) -> Result<Value> {
    match &args[0] {
        Value::Instance(inst) => {
            if let Value::String(name) = &args[1] {
                let inst = inst.borrow();
                if let Some(value) = inst.fields.get(&**name) {
                    return Ok(value.clone());
                }
                if let Some(method) = inst.class.methods.get(&**name) {
                    return Ok(method.clone());
                }
                if args.len() > 2 {
                    return Ok(args[2].clone());
                }
                Err(Error::Attribute(format!(
                    "'{}' object has no attribute '{}'",
                    inst.class.name, name
                )))
            } else {
                Err(Error::Type("getattr() attribute name must be string".into()))
            }
        }
        _ => {
            if args.len() > 2 {
                Ok(args[2].clone())
            } else {
                Err(Error::Attribute("object has no attribute".into()))
            }
        }
    }
}

fn builtin_setattr(args: &[Value]) -> Result<Value> {
    match &args[0] {
        Value::Instance(inst) => {
            if let Value::String(name) = &args[1] {
                inst.borrow_mut().fields.insert((**name).clone(), args[2].clone());
                Ok(Value::None)
            } else {
                Err(Error::Type("setattr() attribute name must be string".into()))
            }
        }
        _ => Err(Error::Type("setattr() requires object".into())),
    }
}

// Helper functions
fn format_value(value: &Value) -> String {
    match value {
        Value::None => "None".to_string(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => {
            if f.fract() == 0.0 {
                format!("{}.0", f)
            } else {
                f.to_string()
            }
        }
        Value::String(s) => (**s).clone(),
        Value::List(l) => {
            let items: Vec<String> = l.borrow().iter().map(|v| repr_value(v)).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Dict(d) => {
            let items: Vec<String> = d
                .borrow()
                .iter()
                .map(|(k, v)| format!("'{}': {}", k, repr_value(v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        Value::Tuple(t) => {
            let items: Vec<String> = t.iter().map(|v| repr_value(v)).collect();
            if items.len() == 1 {
                format!("({},)", items[0])
            } else {
                format!("({})", items.join(", "))
            }
        }
        Value::Function(f) => format!("<function {}>", f.name),
        Value::NativeFunction(f) => format!("<built-in function {}>", f.name),
        Value::Class(c) => format!("<class '{}'>", c.name),
        Value::Instance(i) => format!("<{} object>", i.borrow().class.name),
        Value::Range(r) => format!("range({}, {}, {})", r.start, r.stop, r.step),
        _ => "<object>".to_string(),
    }
}

fn repr_value(value: &Value) -> String {
    match value {
        Value::String(s) => format!("'{}'", s),
        _ => format_value(value),
    }
}

fn find_min(values: &[Value]) -> Result<Value> {
    let mut min = values[0].clone();
    for v in &values[1..] {
        match (&min, v) {
            (Value::Int(a), Value::Int(b)) if b < a => min = v.clone(),
            (Value::Float(a), Value::Float(b)) if b < a => min = v.clone(),
            (Value::String(a), Value::String(b)) if b < a => min = v.clone(),
            _ => {}
        }
    }
    Ok(min)
}

fn find_max(values: &[Value]) -> Result<Value> {
    let mut max = values[0].clone();
    for v in &values[1..] {
        match (&max, v) {
            (Value::Int(a), Value::Int(b)) if b > a => max = v.clone(),
            (Value::Float(a), Value::Float(b)) if b > a => max = v.clone(),
            (Value::String(a), Value::String(b)) if b > a => max = v.clone(),
            _ => {}
        }
    }
    Ok(max)
}

// ============================================================================
// Decorator Built-ins
// ============================================================================

/// The @property decorator.
///
/// Creates a property descriptor that intercepts attribute access.
/// Usage:
///   @property
///   def name(self):
///       return self._name
///
///   @name.setter
///   def name(self, value):
///       self._name = value
fn builtin_property(args: &[Value]) -> Result<Value> {
    let fget = if args.is_empty() {
        None
    } else {
        Some(args[0].clone())
    };

    Ok(Value::Property(Rc::new(PropertyDescriptor::new(fget))))
}

/// The @staticmethod decorator.
///
/// Creates a static method that doesn't receive self or cls.
/// Usage:
///   @staticmethod
///   def create():
///       return MyClass()
fn builtin_staticmethod(args: &[Value]) -> Result<Value> {
    if args.is_empty() {
        return Err(Error::Type("staticmethod() requires a function argument".into()));
    }

    Ok(Value::StaticMethod(Rc::new(StaticMethodDescriptor::new(
        args[0].clone(),
    ))))
}

/// The @classmethod decorator.
///
/// Creates a class method that receives the class as first argument.
/// Usage:
///   @classmethod
///   def from_string(cls, s):
///       return cls(int(s))
fn builtin_classmethod(args: &[Value]) -> Result<Value> {
    if args.is_empty() {
        return Err(Error::Type("classmethod() requires a function argument".into()));
    }

    Ok(Value::ClassMethod(Rc::new(ClassMethodDescriptor::new(
        args[0].clone(),
    ))))
}
