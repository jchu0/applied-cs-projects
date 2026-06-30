//! Virtual machine for bytecode execution.

use crate::builtins;
use crate::compiler::{CodeObject, FunctionType, OpCode};
use crate::value::*;
use crate::{Error, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Exception handler.
#[derive(Clone)]
struct ExceptionHandler {
    handler_ip: usize,  // IP to jump to on exception
    stack_level: usize, // Stack size when handler was set
    frame_idx: usize,   // Frame index when handler was set
}

/// Call frame representing a function invocation context.
///
/// Key design: Local variables are stored separately from the expression stack to prevent
/// for-loop iteration from corrupting the iterator. Previously, StoreFast would write to
/// `stack[bp + idx]`, which could overwrite values pushed for iteration. Now locals live
/// in their own vector, keeping the expression stack clean for temporaries and iterators.
struct CallFrame {
    code: Rc<CodeObject>,
    ip: usize,
    bp: usize,
    /// Local variables indexed by varname slot. Separate from expression stack to avoid
    /// corrupting iterators and temporaries during for-loop variable assignment.
    locals: Vec<Value>,
    /// Closure cells for accessing variables from enclosing scopes (free variables).
    closure: Option<Rc<RefCell<Vec<Value>>>>,
    /// When calling __init__, this holds the instance to return instead of None.
    /// Python's __init__ returns None but the constructor should return the instance.
    returns_instance: Option<Value>,
}

/// Virtual machine.
pub struct VM {
    stack: Vec<Value>,
    frames: Vec<CallFrame>,
    globals: HashMap<String, Value>,
    exception_handlers: Vec<ExceptionHandler>,
    current_exception: Option<Value>,
}

impl VM {
    /// Create a new VM.
    pub fn new() -> Self {
        let mut globals = HashMap::new();
        builtins::register_builtins(&mut globals);

        Self {
            stack: Vec::new(),
            frames: Vec::new(),
            globals,
            exception_handlers: Vec::new(),
            current_exception: None,
        }
    }

    /// Run bytecode.
    pub fn run(&mut self, code: &Rc<CodeObject>) -> Result<Value> {
        self.frames.push(CallFrame {
            code: code.clone(),
            ip: 0,
            bp: 0,
            locals: Vec::new(),
            closure: None,
            returns_instance: None,
        });

        self.execute()
    }

    fn execute(&mut self) -> Result<Value> {
        loop {
            let op = self.read_op();

            match op {
                OpCode::LoadConst => {
                    let idx = self.read_u16();
                    let value = self.current_frame().code.constants[idx as usize].clone();
                    self.push(value);
                }

                OpCode::LoadName => {
                    let idx = self.read_u16();
                    let name = &self.current_frame().code.names[idx as usize];
                    let value = self.globals.get(name).cloned().ok_or_else(|| {
                        Error::Name(format!("name '{}' is not defined", name))
                    })?;
                    self.push(value);
                }

                OpCode::StoreName => {
                    let idx = self.read_u16();
                    let name = self.current_frame().code.names[idx as usize].clone();
                    let value = self.pop();
                    self.globals.insert(name, value);
                }

                // Load local variable from frame.locals (not stack-relative).
                // This allows for-loops to work correctly since the iterator stays on the
                // expression stack while loop variables are stored in the locals vector.
                OpCode::LoadFast => {
                    let idx = self.read_u16() as usize;
                    let frame = self.frames.last().unwrap();
                    let value = frame.locals.get(idx).cloned().unwrap_or(Value::None);
                    self.push(value);
                }

                // Store local variable to frame.locals (not stack-relative).
                // Critical fix: Previously storing at stack[bp + idx] would overwrite the
                // iterator on the expression stack during for-loop variable assignment.
                OpCode::StoreFast => {
                    let idx = self.read_u16() as usize;
                    let value = self.pop();
                    let frame = self.frames.last_mut().unwrap();
                    if idx >= frame.locals.len() {
                        frame.locals.resize(idx + 1, Value::None);
                    }
                    frame.locals[idx] = value;
                }

                OpCode::LoadAttr => {
                    let idx = self.read_u16();
                    let name = self.current_frame().code.names[idx as usize].clone();
                    let obj = self.pop();
                    let value = self.get_attr(&obj, &name)?;
                    self.push(value);
                }

                OpCode::StoreAttr => {
                    let idx = self.read_u16();
                    let name = self.current_frame().code.names[idx as usize].clone();
                    let obj = self.pop();
                    let value = self.pop();
                    self.set_attr(obj, &name, value)?;
                }

                OpCode::LoadSubscript => {
                    let index = self.pop();
                    let obj = self.pop();
                    let value = self.get_subscript(&obj, &index)?;
                    self.push(value);
                }

                OpCode::StoreSubscript => {
                    let index = self.pop();
                    let obj = self.pop();
                    let value = self.pop();
                    self.set_subscript(obj, index, value)?;
                }

                // Arithmetic
                OpCode::BinaryAdd => {
                    let b = self.pop();
                    let a = self.pop();
                    let result = self.add(&a, &b)?;
                    self.push(result);
                }

                OpCode::BinarySub => {
                    let b = self.pop();
                    let a = self.pop();
                    let result = self.sub(&a, &b)?;
                    self.push(result);
                }

                OpCode::BinaryMul => {
                    let b = self.pop();
                    let a = self.pop();
                    let result = self.mul(&a, &b)?;
                    self.push(result);
                }

                OpCode::BinaryDiv => {
                    let b = self.pop();
                    let a = self.pop();
                    let result = self.div(&a, &b)?;
                    self.push(result);
                }

                OpCode::BinaryFloorDiv => {
                    let b = self.pop();
                    let a = self.pop();
                    let result = self.floor_div(&a, &b)?;
                    self.push(result);
                }

                OpCode::BinaryMod => {
                    let b = self.pop();
                    let a = self.pop();
                    let result = self.modulo(&a, &b)?;
                    self.push(result);
                }

                OpCode::BinaryPow => {
                    let b = self.pop();
                    let a = self.pop();
                    let result = self.pow(&a, &b)?;
                    self.push(result);
                }

                OpCode::UnaryNeg => {
                    let a = self.pop();
                    let result = match a {
                        Value::Int(i) => Value::Int(-i),
                        Value::Float(f) => Value::Float(-f),
                        _ => return Err(Error::Type("unary - requires numeric".into())),
                    };
                    self.push(result);
                }

                OpCode::UnaryNot => {
                    let a = self.pop();
                    self.push(Value::Bool(!a.is_truthy()));
                }

                // Comparison
                OpCode::CompareEq => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Bool(a == b));
                }

                OpCode::CompareNe => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Bool(a != b));
                }

                OpCode::CompareLt => {
                    let b = self.pop();
                    let a = self.pop();
                    let result = self.compare_lt(&a, &b)?;
                    self.push(Value::Bool(result));
                }

                OpCode::CompareLe => {
                    let b = self.pop();
                    let a = self.pop();
                    let result = self.compare_le(&a, &b)?;
                    self.push(Value::Bool(result));
                }

                OpCode::CompareGt => {
                    let b = self.pop();
                    let a = self.pop();
                    let result = self.compare_lt(&b, &a)?;
                    self.push(Value::Bool(result));
                }

                OpCode::CompareGe => {
                    let b = self.pop();
                    let a = self.pop();
                    let result = self.compare_le(&b, &a)?;
                    self.push(Value::Bool(result));
                }

                OpCode::CompareIs => {
                    let b = self.pop();
                    let a = self.pop();
                    // Simplified - just check equality
                    self.push(Value::Bool(std::ptr::eq(&a, &b)));
                }

                OpCode::CompareIn => {
                    let container = self.pop();
                    let item = self.pop();
                    let result = self.contains(&container, &item)?;
                    self.push(Value::Bool(result));
                }

                // Control flow
                OpCode::Jump => {
                    let offset = self.read_i16();
                    self.jump(offset);
                }

                OpCode::JumpIfTrue => {
                    let offset = self.read_i16();
                    if self.peek().is_truthy() {
                        self.jump(offset);
                    }
                }

                OpCode::JumpIfFalse => {
                    let offset = self.read_i16();
                    if !self.peek().is_truthy() {
                        self.jump(offset);
                    }
                }

                OpCode::PopJumpIfTrue => {
                    let offset = self.read_i16();
                    let value = self.pop();
                    if value.is_truthy() {
                        self.jump(offset);
                    }
                }

                OpCode::PopJumpIfFalse => {
                    let offset = self.read_i16();
                    let value = self.pop();
                    if !value.is_truthy() {
                        self.jump(offset);
                    }
                }

                // Functions
                OpCode::Call => {
                    let arg_count = self.read_u8() as usize;
                    let callee = self.stack[self.stack.len() - arg_count - 1].clone();

                    match callee {
                        Value::Function(func) => {
                            if arg_count != func.code.arg_count as usize {
                                return Err(Error::Type(format!(
                                    "{}() takes {} arguments but {} were given",
                                    func.name, func.code.arg_count, arg_count
                                )));
                            }

                            // Check if this is a generator or coroutine function
                            match func.code.function_type {
                                FunctionType::Generator => {
                                    // Create a generator object instead of executing
                                    let args: Vec<Value> =
                                        self.stack.drain(self.stack.len() - arg_count..).collect();
                                    self.pop(); // Pop function

                                    let mut gen = Generator::new(func.name.clone(), func.code.clone());
                                    // Store arguments in generator's locals
                                    for (i, arg) in args.into_iter().enumerate() {
                                        if let Some(name) = func.code.varnames.get(i) {
                                            gen.locals.insert(name.clone(), arg);
                                        }
                                    }
                                    self.push(Value::Generator(Rc::new(RefCell::new(gen))));
                                    continue;
                                }
                                FunctionType::Coroutine | FunctionType::AsyncGenerator => {
                                    // Create a coroutine object instead of executing
                                    let args: Vec<Value> =
                                        self.stack.drain(self.stack.len() - arg_count..).collect();
                                    self.pop(); // Pop function

                                    let mut coro = Coroutine::new(func.name.clone(), func.code.clone());
                                    // Store arguments in coroutine's locals
                                    for (i, arg) in args.into_iter().enumerate() {
                                        if let Some(name) = func.code.varnames.get(i) {
                                            coro.locals.insert(name.clone(), arg);
                                        }
                                    }
                                    self.push(Value::Coroutine(Rc::new(RefCell::new(coro))));
                                    continue;
                                }
                                FunctionType::Regular => {
                                    // Normal function call - args are on stack
                                    // Copy args to locals vector
                                    let args: Vec<Value> =
                                        self.stack.drain(self.stack.len() - arg_count..).collect();
                                    self.pop(); // Pop function

                                    self.frames.push(CallFrame {
                                        code: func.code.clone(),
                                        ip: 0,
                                        bp: self.stack.len(),
                                        locals: args,  // Arguments become locals
                                        closure: func.closure.clone(),
                                        returns_instance: None,
                                    });
                                }
                            }
                        }
                        Value::NativeFunction(func) => {
                            let args: Vec<Value> =
                                self.stack.drain(self.stack.len() - arg_count..).collect();
                            self.pop(); // Pop function
                            let result = (func.func)(&args)?;
                            self.push(result);
                        }
                        Value::Class(class) => {
                            // Create instance
                            let instance = Instance {
                                class: class.clone(),
                                fields: HashMap::new(),
                            };
                            let instance = Value::Instance(Rc::new(RefCell::new(instance)));

                            // Call __init__ if present
                            if let Some(init) = class.methods.get("__init__") {
                                // Pop args
                                let args: Vec<Value> =
                                    self.stack.drain(self.stack.len() - arg_count..).collect();
                                self.pop(); // Pop class

                                // Build locals: [self, *args]
                                let mut locals = vec![instance.clone()];
                                locals.extend(args);

                                if let Value::Function(func) = init {
                                    self.frames.push(CallFrame {
                                        code: func.code.clone(),
                                        ip: 0,
                                        bp: self.stack.len(),
                                        locals,
                                        closure: func.closure.clone(),
                                        returns_instance: Some(instance.clone()),
                                    });
                                    continue;
                                }
                            } else {
                                // Pop args and class
                                self.stack.truncate(self.stack.len() - arg_count - 1);
                                self.push(instance);
                            }
                        }
                        Value::BoundMethod(method) => {
                            if let Value::Function(func) = &method.method {
                                // Pop args and function
                                let args: Vec<Value> =
                                    self.stack.drain(self.stack.len() - arg_count..).collect();
                                self.pop(); // Pop method

                                // Build locals: [self, *args]
                                let mut locals = vec![method.receiver.clone()];
                                locals.extend(args);

                                self.frames.push(CallFrame {
                                    code: func.code.clone(),
                                    ip: 0,
                                    bp: self.stack.len(),
                                    locals,
                                    closure: func.closure.clone(),
                                    returns_instance: None,
                                });
                            }
                        }
                        // Handle list methods like append, pop, extend, etc.
                        // BoundListMethod captures both the list reference and the method name,
                        // allowing us to call list methods as if they were bound methods.
                        Value::BoundListMethod(list, method_name) => {
                            let args: Vec<Value> =
                                self.stack.drain(self.stack.len() - arg_count..).collect();
                            self.pop(); // Pop the method

                            let result = match method_name.as_str() {
                                "append" => {
                                    if args.len() != 1 {
                                        return Err(Error::Type("append() takes exactly 1 argument".into()));
                                    }
                                    list.borrow_mut().push(args[0].clone());
                                    Value::None
                                }
                                "pop" => {
                                    if args.len() > 1 {
                                        return Err(Error::Type("pop() takes at most 1 argument".into()));
                                    }
                                    let mut l = list.borrow_mut();
                                    if args.is_empty() {
                                        l.pop().ok_or_else(|| Error::Runtime("pop from empty list".into()))?
                                    } else if let Value::Int(i) = &args[0] {
                                        let idx = if *i < 0 {
                                            (l.len() as i64 + *i) as usize
                                        } else {
                                            *i as usize
                                        };
                                        if idx < l.len() {
                                            l.remove(idx)
                                        } else {
                                            return Err(Error::Runtime("pop index out of range".into()));
                                        }
                                    } else {
                                        return Err(Error::Type("pop index must be an integer".into()));
                                    }
                                }
                                "extend" => {
                                    if args.len() != 1 {
                                        return Err(Error::Type("extend() takes exactly 1 argument".into()));
                                    }
                                    if let Value::List(other) = &args[0] {
                                        list.borrow_mut().extend(other.borrow().iter().cloned());
                                        Value::None
                                    } else {
                                        return Err(Error::Type("extend() argument must be a list".into()));
                                    }
                                }
                                "insert" => {
                                    if args.len() != 2 {
                                        return Err(Error::Type("insert() takes exactly 2 arguments".into()));
                                    }
                                    if let Value::Int(i) = &args[0] {
                                        let mut l = list.borrow_mut();
                                        let idx = if *i < 0 {
                                            0.max((l.len() as i64 + *i) as usize)
                                        } else {
                                            (*i as usize).min(l.len())
                                        };
                                        l.insert(idx, args[1].clone());
                                        Value::None
                                    } else {
                                        return Err(Error::Type("insert index must be an integer".into()));
                                    }
                                }
                                "remove" => {
                                    if args.len() != 1 {
                                        return Err(Error::Type("remove() takes exactly 1 argument".into()));
                                    }
                                    let mut l = list.borrow_mut();
                                    if let Some(pos) = l.iter().position(|x| x == &args[0]) {
                                        l.remove(pos);
                                        Value::None
                                    } else {
                                        return Err(Error::Runtime("value not in list".into()));
                                    }
                                }
                                "clear" => {
                                    list.borrow_mut().clear();
                                    Value::None
                                }
                                _ => return Err(Error::Type(format!("unknown list method '{}'", method_name))),
                            };
                            self.push(result);
                        }
                        _ => {
                            return Err(Error::Type(format!(
                                "'{}' object is not callable",
                                callee.type_name()
                            )));
                        }
                    }
                }

                // Return from function. For __init__ methods, we return the instance
                // (stored in returns_instance) rather than the actual return value (None).
                // This ensures `obj = MyClass()` gets the instance, not None.
                OpCode::Return => {
                    let result = self.pop();
                    let frame = self.frames.pop().unwrap();

                    // For __init__, return the instance instead of the function's return value.
                    // Python's __init__ implicitly returns None, but the class call should
                    // return the newly created instance.
                    let _is_init = frame.returns_instance.is_some();
                    let final_result = frame.returns_instance.unwrap_or(result);

                    if self.frames.is_empty() {
                        return Ok(final_result);
                    }

                    // Clean up expression stack. Locals are in frame.locals, not the stack,
                    // so we just truncate back to the base pointer.
                    self.stack.truncate(frame.bp);
                    self.push(final_result);
                }

                OpCode::MakeFunction => {
                    // Function is already on stack as constant
                }

                // Collections
                OpCode::BuildList => {
                    let count = self.read_u16() as usize;
                    let elements: Vec<Value> =
                        self.stack.drain(self.stack.len() - count..).collect();
                    self.push(Value::List(Rc::new(RefCell::new(elements))));
                }

                OpCode::BuildDict => {
                    let count = self.read_u16() as usize;
                    let mut dict = HashMap::new();
                    let items: Vec<Value> =
                        self.stack.drain(self.stack.len() - count * 2..).collect();
                    for chunk in items.chunks(2) {
                        if let Value::String(key) = &chunk[0] {
                            dict.insert((**key).clone(), chunk[1].clone());
                        }
                    }
                    self.push(Value::Dict(Rc::new(RefCell::new(dict))));
                }

                OpCode::BuildTuple => {
                    let count = self.read_u16() as usize;
                    let elements: Vec<Value> =
                        self.stack.drain(self.stack.len() - count..).collect();
                    self.push(Value::Tuple(Rc::new(elements)));
                }

                OpCode::ListAppend => {
                    let item = self.pop();
                    let list = self.pop();
                    if let Value::List(l) = list {
                        l.borrow_mut().push(item);
                        self.push(Value::List(l));
                    }
                }

                // Iteration
                OpCode::GetIter => {
                    let value = self.pop();
                    let iter = self.make_iterator(value)?;
                    self.push(iter);
                }

                OpCode::ForIter => {
                    let offset = self.read_i16();
                    let iter = self.peek().clone();

                    if let Value::Iterator(it) = iter {
                        if let Some(value) = it.borrow_mut().next_value() {
                            self.push(value);
                        } else {
                            self.jump(offset);
                        }
                    } else {
                        return Err(Error::Type("not an iterator".into()));
                    }
                }

                // Class
                OpCode::BuildClass => {
                    let name = self.pop();
                    let code = self.pop();
                    let bases = self.pop();

                    let class_name = match name {
                        Value::String(s) => (*s).clone(),
                        _ => "<class>".into(),
                    };

                    // Execute class body to get methods
                    let methods = if let Value::Function(func) = code {
                        // Run class body in a new VM with access to outer globals
                        let mut class_vm = VM::new();
                        // Copy globals from outer VM so decorators and other names are visible
                        class_vm.globals = self.globals.clone();
                        class_vm.run(&func.code)?;
                        // Return only the new definitions (methods), not the inherited globals
                        let mut methods = HashMap::new();
                        for (k, v) in class_vm.globals {
                            // Only include values defined in class body (not pre-existing globals)
                            if !self.globals.contains_key(&k) {
                                methods.insert(k, v);
                            }
                        }
                        methods
                    } else {
                        HashMap::new()
                    };

                    let class = Class {
                        name: class_name,
                        bases: Vec::new(),
                        methods,
                    };

                    self.push(Value::Class(Rc::new(class)));
                }

                OpCode::LoadMethod => {
                    let idx = self.read_u16();
                    let name = self.current_frame().code.names[idx as usize].clone();
                    let obj = self.pop();
                    let method = self.get_attr(&obj, &name)?;

                    // Create bound method
                    let bound = BoundMethod {
                        receiver: obj,
                        method,
                    };
                    self.push(Value::BoundMethod(Rc::new(bound)));
                }

                OpCode::CallMethod => {
                    let arg_count = self.read_u8();
                    // Handled same as Call
                }

                // Misc
                OpCode::Pop => {
                    self.pop();
                }

                OpCode::Dup => {
                    let value = self.peek().clone();
                    self.push(value);
                }

                OpCode::Nop => {}

                OpCode::RotTwo => {
                    let len = self.stack.len();
                    if len >= 2 {
                        self.stack.swap(len - 1, len - 2);
                    }
                }

                // Exception handling
                OpCode::SetupExcept => {
                    let offset = self.read_i16();
                    let handler_ip = (self.current_frame().ip as i32 + offset as i32) as usize;
                    self.exception_handlers.push(ExceptionHandler {
                        handler_ip,
                        stack_level: self.stack.len(),
                        frame_idx: self.frames.len() - 1,
                    });
                }

                OpCode::PopExcept => {
                    self.exception_handlers.pop();
                }

                OpCode::Raise => {
                    let exc_value = self.pop();
                    self.current_exception = Some(exc_value.clone());

                    // Find exception handler
                    if let Some(handler) = self.exception_handlers.pop() {
                        // Unwind stack to handler's level
                        self.stack.truncate(handler.stack_level);

                        // Unwind frames to handler's frame
                        while self.frames.len() > handler.frame_idx + 1 {
                            self.frames.pop();
                        }

                        // Jump to handler
                        self.current_frame_mut().ip = handler.handler_ip;

                        // Push exception value for handler
                        self.push(exc_value);
                    } else {
                        // No handler - propagate as error
                        return Err(Error::Runtime(format!(
                            "Unhandled exception: {}",
                            exc_value.repr()
                        )));
                    }
                }

                OpCode::Reraise => {
                    if let Some(exc) = self.current_exception.clone() {
                        // Find exception handler
                        if let Some(handler) = self.exception_handlers.pop() {
                            self.stack.truncate(handler.stack_level);
                            while self.frames.len() > handler.frame_idx + 1 {
                                self.frames.pop();
                            }
                            self.current_frame_mut().ip = handler.handler_ip;
                            self.push(exc);
                        } else {
                            return Err(Error::Runtime(format!(
                                "Unhandled exception: {}",
                                exc.repr()
                            )));
                        }
                    } else {
                        return Err(Error::Runtime(
                            "No active exception to re-raise".into(),
                        ));
                    }
                }

                OpCode::EndFinally => {
                    // Clear current exception after finally block
                    self.current_exception = None;
                }

                // Generator/coroutine opcodes
                OpCode::YieldValue => {
                    // This is handled specially during generator iteration
                    // When executing a generator frame, this causes suspension
                    let value = self.pop();
                    return Ok(value);
                }

                OpCode::YieldFrom => {
                    // Delegate to sub-iterator
                    let iter = self.pop();
                    if let Value::Iterator(it) = iter {
                        if let Some(value) = it.borrow_mut().next_value() {
                            self.push(value);
                        } else {
                            let idx = self.add_none_constant();
                            self.push(Value::None);
                        }
                    } else if let Value::Generator(gen) = iter {
                        // Run generator to get next value
                        let result = self.run_generator(&gen)?;
                        self.push(result);
                    } else {
                        return Err(Error::Type("yield from requires an iterable".into()));
                    }
                }

                OpCode::GetAwaitable => {
                    let value = self.pop();
                    // For now, treat coroutines as awaitables
                    match value {
                        Value::Coroutine(_) => self.push(value),
                        _ => {
                            // Check for __await__ method
                            self.push(value);
                        }
                    }
                }

                OpCode::GetAiter => {
                    let value = self.pop();
                    // For now, just get regular iterator
                    let iter = self.make_iterator(value)?;
                    self.push(iter);
                }

                OpCode::GetAnext => {
                    let offset = self.read_i16();
                    let iter = self.peek().clone();

                    if let Value::Iterator(it) = iter {
                        if let Some(value) = it.borrow_mut().next_value() {
                            self.push(value);
                        } else {
                            self.jump(offset);
                        }
                    } else {
                        return Err(Error::Type("not an async iterator".into()));
                    }
                }

                OpCode::SetupWith => {
                    let offset = self.read_i16();
                    let context = self.pop();

                    // Call __enter__ method
                    let enter_result = self.call_method(&context, "__enter__", &[])?;
                    self.push(enter_result);

                    // Store context for cleanup
                    self.push(context);
                }

                OpCode::WithCleanup => {
                    let context = self.pop();
                    // Call __exit__ method
                    let _ = self.call_method(&context, "__exit__", &[Value::None, Value::None, Value::None]);
                }

                OpCode::LoadDeref => {
                    // Load value from closure cell
                    let idx = self.read_u16() as usize;
                    let value = if let Some(closure) = &self.current_frame().closure {
                        let cells = closure.borrow();
                        cells.get(idx).cloned().unwrap_or(Value::None)
                    } else {
                        // No closure - try to look up in current frame's locals
                        let frame = self.current_frame();
                        if idx < frame.code.freevars.len() {
                            let name = &frame.code.freevars[idx];
                            // Look up in globals as fallback
                            self.globals.get(name).cloned().unwrap_or(Value::None)
                        } else {
                            Value::None
                        }
                    };
                    self.push(value);
                }

                OpCode::StoreDeref => {
                    // Store value to closure cell
                    let idx = self.read_u16() as usize;
                    let value = self.pop();
                    if let Some(closure) = &self.current_frame().closure {
                        let mut cells = closure.borrow_mut();
                        if idx < cells.len() {
                            cells[idx] = value;
                        }
                    }
                }

                OpCode::LoadClosure => {
                    // LoadClosure is not needed as a separate opcode
                    // MakeClosure handles building the closure from stack values
                }

                OpCode::MakeClosure => {
                    // Create a closure: function + captured values on stack
                    let count = self.read_u16() as usize;

                    // Pop captured values in reverse order
                    let mut cells: Vec<Value> = self.stack.drain(self.stack.len() - count..).collect();
                    // Cells are in correct order now

                    // Pop the function
                    let func = self.pop();

                    if let Value::Function(func_obj) = func {
                        // Create new function with closure
                        let closure_func = crate::value::Function {
                            name: func_obj.name.clone(),
                            code: func_obj.code.clone(),
                            defaults: func_obj.defaults.clone(),
                            closure: Some(Rc::new(RefCell::new(cells))),
                        };
                        self.push(Value::Function(Rc::new(closure_func)));
                    } else {
                        // Shouldn't happen, but push back as-is
                        self.push(func);
                    }
                }

                _ => {
                    return Err(Error::Runtime(format!("Unknown opcode: {:?}", op)));
                }
            }
        }
    }

    // Stack operations
    fn push(&mut self, value: Value) {
        self.stack.push(value);
    }

    fn pop(&mut self) -> Value {
        self.stack.pop().unwrap_or(Value::None)
    }

    fn peek(&self) -> &Value {
        self.stack.last().unwrap_or(&Value::None)
    }

    // Frame operations
    fn current_frame(&self) -> &CallFrame {
        self.frames.last().unwrap()
    }

    fn current_frame_mut(&mut self) -> &mut CallFrame {
        self.frames.last_mut().unwrap()
    }

    fn read_op(&mut self) -> OpCode {
        let byte = self.current_frame().code.bytecode[self.current_frame().ip];
        self.current_frame_mut().ip += 1;
        unsafe { std::mem::transmute(byte) }
    }

    fn read_u8(&mut self) -> u8 {
        let byte = self.current_frame().code.bytecode[self.current_frame().ip];
        self.current_frame_mut().ip += 1;
        byte
    }

    fn read_u16(&mut self) -> u16 {
        let high = self.read_u8() as u16;
        let low = self.read_u8() as u16;
        (high << 8) | low
    }

    fn read_i16(&mut self) -> i16 {
        self.read_u16() as i16
    }

    fn jump(&mut self, offset: i16) {
        let new_ip = (self.current_frame().ip as i32 + offset as i32) as usize;
        self.current_frame_mut().ip = new_ip;
    }

    // Operations
    fn add(&self, a: &Value, b: &Value) -> Result<Value> {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x + y)),
            (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x + y)),
            (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 + y)),
            (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x + *y as f64)),
            (Value::String(x), Value::String(y)) => {
                Ok(Value::String(Rc::new(format!("{}{}", x, y))))
            }
            (Value::List(x), Value::List(y)) => {
                let mut result = x.borrow().clone();
                result.extend(y.borrow().clone());
                Ok(Value::List(Rc::new(RefCell::new(result))))
            }
            _ => Err(Error::Type(format!(
                "unsupported operand type(s) for +: '{}' and '{}'",
                a.type_name(),
                b.type_name()
            ))),
        }
    }

    fn sub(&self, a: &Value, b: &Value) -> Result<Value> {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x - y)),
            (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x - y)),
            (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 - y)),
            (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x - *y as f64)),
            _ => Err(Error::Type("unsupported operand type(s) for -".into())),
        }
    }

    fn mul(&self, a: &Value, b: &Value) -> Result<Value> {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x * y)),
            (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x * y)),
            (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 * y)),
            (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x * *y as f64)),
            (Value::String(s), Value::Int(n)) | (Value::Int(n), Value::String(s)) => {
                Ok(Value::String(Rc::new(s.repeat(*n as usize))))
            }
            _ => Err(Error::Type("unsupported operand type(s) for *".into())),
        }
    }

    fn div(&self, a: &Value, b: &Value) -> Result<Value> {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => {
                if *y == 0 {
                    Err(Error::Runtime("division by zero".into()))
                } else {
                    Ok(Value::Float(*x as f64 / *y as f64))
                }
            }
            (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x / y)),
            (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 / y)),
            (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x / *y as f64)),
            _ => Err(Error::Type("unsupported operand type(s) for /".into())),
        }
    }

    fn floor_div(&self, a: &Value, b: &Value) -> Result<Value> {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => {
                if *y == 0 {
                    Err(Error::Runtime("division by zero".into()))
                } else {
                    Ok(Value::Int(x / y))
                }
            }
            _ => Err(Error::Type("unsupported operand type(s) for //".into())),
        }
    }

    fn modulo(&self, a: &Value, b: &Value) -> Result<Value> {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => {
                if *y == 0 {
                    Err(Error::Runtime("modulo by zero".into()))
                } else {
                    Ok(Value::Int(x % y))
                }
            }
            _ => Err(Error::Type("unsupported operand type(s) for %".into())),
        }
    }

    fn pow(&self, a: &Value, b: &Value) -> Result<Value> {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => {
                if *y >= 0 {
                    Ok(Value::Int(x.pow(*y as u32)))
                } else {
                    Ok(Value::Float((*x as f64).powf(*y as f64)))
                }
            }
            (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x.powf(*y))),
            (Value::Int(x), Value::Float(y)) => Ok(Value::Float((*x as f64).powf(*y))),
            (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x.powf(*y as f64))),
            _ => Err(Error::Type("unsupported operand type(s) for **".into())),
        }
    }

    fn compare_lt(&self, a: &Value, b: &Value) -> Result<bool> {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => Ok(x < y),
            (Value::Float(x), Value::Float(y)) => Ok(x < y),
            (Value::Int(x), Value::Float(y)) => Ok((*x as f64) < *y),
            (Value::Float(x), Value::Int(y)) => Ok(*x < *y as f64),
            (Value::String(x), Value::String(y)) => Ok(x < y),
            _ => Err(Error::Type("not supported between instances".into())),
        }
    }

    fn compare_le(&self, a: &Value, b: &Value) -> Result<bool> {
        Ok(self.compare_lt(a, b)? || a == b)
    }

    fn contains(&self, container: &Value, item: &Value) -> Result<bool> {
        match container {
            Value::List(l) => Ok(l.borrow().contains(item)),
            Value::Tuple(t) => Ok(t.contains(item)),
            Value::String(s) => {
                if let Value::String(sub) = item {
                    Ok(s.contains(&**sub))
                } else {
                    Err(Error::Type("'in' requires string".into()))
                }
            }
            Value::Dict(d) => {
                if let Value::String(key) = item {
                    Ok(d.borrow().contains_key(&**key))
                } else {
                    Err(Error::Type("dict key must be string".into()))
                }
            }
            _ => Err(Error::Type("argument of type is not iterable".into())),
        }
    }

    fn get_attr(&self, obj: &Value, name: &str) -> Result<Value> {
        match obj {
            Value::Instance(inst) => {
                // First check instance fields
                {
                    let inst_borrowed = inst.borrow();
                    if let Some(value) = inst_borrowed.fields.get(name) {
                        return Ok(value.clone());
                    }
                }

                // Then check class methods/descriptors
                // We need to extract what we need from the borrow, then drop it
                let method_info: Option<(Value, Rc<Class>)> = {
                    let inst_borrowed = inst.borrow();
                    inst_borrowed.class.methods.get(name).map(|m| {
                        (m.clone(), inst_borrowed.class.clone())
                    })
                };

                if let Some((method, class)) = method_info {
                    // Handle descriptor protocol
                    match &method {
                        // Property: call the getter
                        Value::Property(prop) => {
                            if let Some(ref fget) = prop.fget {
                                let receiver = Value::Instance(Rc::clone(inst));
                                let bound = BoundMethod {
                                    receiver,
                                    method: fget.clone(),
                                };
                                return Ok(Value::BoundMethod(Rc::new(bound)));
                            }
                            return Err(Error::Attribute(format!(
                                "property '{}' has no getter",
                                name
                            )));
                        }
                        // StaticMethod: return the unwrapped function (no binding)
                        Value::StaticMethod(sm) => {
                            return Ok(sm.func.clone());
                        }
                        // ClassMethod: bind to the class
                        Value::ClassMethod(cm) => {
                            let class_value = Value::Class(class);
                            let bound = BoundMethod {
                                receiver: class_value,
                                method: cm.func.clone(),
                            };
                            return Ok(Value::BoundMethod(Rc::new(bound)));
                        }
                        // Regular method: bind to instance
                        _ => {
                            let receiver = Value::Instance(Rc::clone(inst));
                            let bound = BoundMethod {
                                receiver,
                                method: method.clone(),
                            };
                            return Ok(Value::BoundMethod(Rc::new(bound)));
                        }
                    }
                }

                let class_name = inst.borrow().class.name.clone();
                Err(Error::Attribute(format!(
                    "'{}' object has no attribute '{}'",
                    class_name, name
                )))
            }
            Value::Class(class) => {
                if let Some(method) = class.methods.get(name) {
                    // Handle descriptors at class level
                    match method {
                        Value::StaticMethod(sm) => Ok(sm.func.clone()),
                        Value::ClassMethod(cm) => {
                            let bound = BoundMethod {
                                receiver: Value::Class(class.clone()),
                                method: cm.func.clone(),
                            };
                            Ok(Value::BoundMethod(Rc::new(bound)))
                        }
                        _ => Ok(method.clone()),
                    }
                } else {
                    Err(Error::Attribute(format!(
                        "type object '{}' has no attribute '{}'",
                        class.name, name
                    )))
                }
            }
            // List attribute access returns BoundListMethod for supported methods.
            // This allows `my_list.append(x)` to work by capturing both the list
            // and method name, then executing the method when called.
            Value::List(list) => {
                match name {
                    "append" | "pop" | "extend" | "insert" | "remove" | "clear" => {
                        Ok(Value::BoundListMethod(Rc::clone(&list), name.to_string()))
                    }
                    _ => Err(Error::Attribute(format!(
                        "'list' object has no attribute '{}'",
                        name
                    ))),
                }
            }
            _ => Err(Error::Attribute(format!(
                "'{}' object has no attribute '{}'",
                obj.type_name(),
                name
            ))),
        }
    }

    fn set_attr(&mut self, obj: Value, name: &str, value: Value) -> Result<()> {
        match obj {
            Value::Instance(inst) => {
                // Check if there's a property setter in the class
                let maybe_setter = {
                    let inst_borrowed = inst.borrow();
                    if let Some(Value::Property(prop)) = inst_borrowed.class.methods.get(name) {
                        prop.fset.clone()
                    } else {
                        None
                    }
                };

                if let Some(fset) = maybe_setter {
                    // Call the property setter
                    // For now, just set the field directly
                    // A full implementation would call fset(self, value)
                    inst.borrow_mut().fields.insert(name.to_string(), value);
                    Ok(())
                } else {
                    // Regular attribute set
                    inst.borrow_mut().fields.insert(name.to_string(), value);
                    Ok(())
                }
            }
            Value::Class(class) => {
                // Set class attribute (method or class variable)
                // Need to create a mutable copy since Rc doesn't allow mutation
                let mut new_methods = class.methods.clone();
                new_methods.insert(name.to_string(), value);

                // Create new class with updated methods
                let new_class = Rc::new(crate::value::Class {
                    name: class.name.clone(),
                    bases: class.bases.clone(),
                    methods: new_methods,
                });

                // Update globals if the class is stored there
                for (_key, val) in &mut self.globals {
                    if let Value::Class(c) = val {
                        if Rc::ptr_eq(c, &class) {
                            *val = Value::Class(new_class.clone());
                        }
                    }
                }

                Ok(())
            }
            _ => Err(Error::Attribute("cannot set attribute".into())),
        }
    }

    fn get_subscript(&self, obj: &Value, index: &Value) -> Result<Value> {
        match (obj, index) {
            (Value::List(l), Value::Int(i)) => {
                let list = l.borrow();
                let idx = if *i < 0 {
                    (list.len() as i64 + i) as usize
                } else {
                    *i as usize
                };
                list.get(idx)
                    .cloned()
                    .ok_or_else(|| Error::Index("list index out of range".into()))
            }
            (Value::Tuple(t), Value::Int(i)) => {
                let idx = if *i < 0 {
                    (t.len() as i64 + i) as usize
                } else {
                    *i as usize
                };
                t.get(idx)
                    .cloned()
                    .ok_or_else(|| Error::Index("tuple index out of range".into()))
            }
            (Value::String(s), Value::Int(i)) => {
                let chars: Vec<char> = s.chars().collect();
                let idx = if *i < 0 {
                    (chars.len() as i64 + i) as usize
                } else {
                    *i as usize
                };
                chars
                    .get(idx)
                    .map(|c| Value::String(Rc::new(c.to_string())))
                    .ok_or_else(|| Error::Index("string index out of range".into()))
            }
            (Value::Dict(d), Value::String(key)) => d
                .borrow()
                .get(&**key)
                .cloned()
                .ok_or_else(|| Error::Key(key.to_string())),
            _ => Err(Error::Type("not subscriptable".into())),
        }
    }

    fn set_subscript(&mut self, obj: Value, index: Value, value: Value) -> Result<()> {
        match (obj, index) {
            (Value::List(l), Value::Int(i)) => {
                let mut list = l.borrow_mut();
                let idx = if i < 0 {
                    (list.len() as i64 + i) as usize
                } else {
                    i as usize
                };
                if idx < list.len() {
                    list[idx] = value;
                    Ok(())
                } else {
                    Err(Error::Index("list assignment index out of range".into()))
                }
            }
            (Value::Dict(d), Value::String(key)) => {
                d.borrow_mut().insert((*key).clone(), value);
                Ok(())
            }
            _ => Err(Error::Type("does not support item assignment".into())),
        }
    }

    fn make_iterator(&self, value: Value) -> Result<Value> {
        let iter: Box<dyn ValueIterator> = match value {
            Value::Range(r) => Box::new(RangeIterator::new(r.start, r.stop, r.step)),
            Value::List(l) => Box::new(ListIterator::new(l)),
            Value::Tuple(t) => Box::new(TupleIterator::new(t)),
            Value::Dict(d) => Box::new(DictIterator::new(&d)),
            Value::String(s) => Box::new(StringIterator::new(&s)),
            Value::Iterator(_) => return Ok(value),
            Value::Generator(gen) => Box::new(GeneratorIterator::new(gen)),
            _ => {
                return Err(Error::Type(format!(
                    "'{}' object is not iterable",
                    value.type_name()
                )));
            }
        };
        Ok(Value::Iterator(Rc::new(RefCell::new(iter))))
    }

    /// Run a generator to get the next value.
    fn run_generator(&mut self, gen: &Rc<RefCell<Generator>>) -> Result<Value> {
        let mut g = gen.borrow_mut();

        if g.is_exhausted() {
            return Err(Error::Runtime("StopIteration".into()));
        }

        // Mark as running
        g.state = GeneratorState::Running;

        // Create a temporary frame for the generator
        // Build locals from saved generator locals
        let mut locals = Vec::new();
        for i in 0..g.code.varnames.len() {
            let name = &g.code.varnames[i];
            let value = g.locals.get(name).cloned().unwrap_or(Value::None);
            locals.push(value);
        }

        // Restore generator's expression stack
        for value in g.stack.drain(..) {
            self.stack.push(value);
        }

        self.frames.push(CallFrame {
            code: g.code.clone(),
            ip: g.ip,
            bp: self.stack.len(),
            locals,
            closure: None,  // Generators don't capture closures this way
            returns_instance: None,
        });

        drop(g); // Release borrow before executing

        // Execute until yield or return
        let result = self.execute();

        // Save state back to generator
        let mut g = gen.borrow_mut();

        if let Some(frame) = self.frames.last() {
            if Rc::ptr_eq(&frame.code, &g.code) {
                g.ip = frame.ip;
                g.state = GeneratorState::Suspended;

                // Save locals back from frame's locals vector
                let varnames: Vec<String> = g.code.varnames.clone();
                for (i, name) in varnames.iter().enumerate() {
                    if i < frame.locals.len() {
                        g.locals.insert(name.clone(), frame.locals[i].clone());
                    }
                }

                // Save expression stack
                let bp = frame.bp;
                if bp < self.stack.len() {
                    g.stack = self.stack[bp..].to_vec();
                }

                self.frames.pop();
                self.stack.truncate(bp);
            }
        } else {
            g.state = GeneratorState::Completed;
        }

        result
    }

    /// Send a value into a generator.
    pub fn send_to_generator(&mut self, gen: &Rc<RefCell<Generator>>, value: Value) -> Result<Value> {
        {
            let mut g = gen.borrow_mut();
            g.send_value = Some(value);
        }
        self.run_generator(gen)
    }

    /// Call a method on an object.
    fn call_method(&mut self, obj: &Value, method_name: &str, args: &[Value]) -> Result<Value> {
        // Get the method
        let method = self.get_attr(obj, method_name)?;

        // Call it
        match method {
            Value::Function(func) => {
                // Build locals: [self, *args]
                let mut locals = vec![obj.clone()];
                locals.extend(args.iter().cloned());

                self.frames.push(CallFrame {
                    code: func.code.clone(),
                    ip: 0,
                    bp: self.stack.len(),
                    locals,
                    closure: func.closure.clone(),
                    returns_instance: None,
                });

                self.execute()
            }
            Value::NativeFunction(func) => {
                let mut all_args = vec![obj.clone()];
                all_args.extend(args.iter().cloned());
                (func.func)(&all_args)
            }
            Value::BoundMethod(bm) => {
                if let Value::Function(func) = &bm.method {
                    // Build locals: [receiver, *args]
                    let mut locals = vec![bm.receiver.clone()];
                    locals.extend(args.iter().cloned());

                    self.frames.push(CallFrame {
                        code: func.code.clone(),
                        ip: 0,
                        bp: self.stack.len(),
                        locals,
                        closure: func.closure.clone(),
                        returns_instance: None,
                    });

                    self.execute()
                } else {
                    Err(Error::Type("method is not callable".into()))
                }
            }
            _ => Err(Error::Attribute(format!(
                "'{}' object has no method '{}'",
                obj.type_name(),
                method_name
            ))),
        }
    }

    /// Add None constant and return its index (helper for YieldFrom).
    fn add_none_constant(&self) -> usize {
        0 // None is typically the first constant
    }

    /// Iterate over a generator, collecting all values.
    pub fn iter_generator(&mut self, gen: &Rc<RefCell<Generator>>) -> Result<Vec<Value>> {
        let mut results = Vec::new();
        loop {
            match self.run_generator(gen) {
                Ok(value) => results.push(value),
                Err(e) => {
                    if e.to_string().contains("StopIteration") {
                        break;
                    }
                    return Err(e);
                }
            }
        }
        Ok(results)
    }
}

impl Default for VM {
    fn default() -> Self {
        Self::new()
    }
}
