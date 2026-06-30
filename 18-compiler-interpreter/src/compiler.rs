//! Bytecode compiler.

use crate::ast::*;
use crate::value::Value;
use crate::{Error, Result};
use std::collections::HashMap;
use std::rc::Rc;

/// Bytecode opcodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OpCode {
    // Stack operations
    LoadConst,
    LoadName,
    LoadFast,
    StoreName,
    StoreFast,
    LoadAttr,
    StoreAttr,
    LoadSubscript,
    StoreSubscript,

    // Arithmetic
    BinaryAdd,
    BinarySub,
    BinaryMul,
    BinaryDiv,
    BinaryFloorDiv,
    BinaryMod,
    BinaryPow,
    UnaryNeg,
    UnaryNot,

    // Comparison
    CompareEq,
    CompareNe,
    CompareLt,
    CompareLe,
    CompareGt,
    CompareGe,
    CompareIs,
    CompareIn,

    // Control flow
    Jump,
    JumpIfTrue,
    JumpIfFalse,
    PopJumpIfTrue,
    PopJumpIfFalse,

    // Functions
    Call,
    Return,
    MakeFunction,

    // Collections
    BuildList,
    BuildDict,
    BuildTuple,
    ListAppend,

    // Iteration
    GetIter,
    ForIter,

    // Class
    BuildClass,
    LoadMethod,
    CallMethod,

    // Misc
    Pop,
    Dup,
    Nop,
    RotTwo,  // Swap top two stack items

    // Exception handling
    SetupExcept,    // Push exception handler (jump offset)
    PopExcept,      // Pop exception handler
    Raise,          // Raise exception
    Reraise,        // Re-raise current exception
    EndFinally,     // End finally block

    // Generator/coroutine
    YieldValue,         // Yield a value from generator
    YieldFrom,          // Yield from sub-generator
    GetAwaitable,       // Get awaitable from object
    GetAiter,           // Get async iterator
    GetAnext,           // Get next from async iterator
    SetupWith,          // Setup with context manager
    WithCleanup,        // Cleanup with context manager

    // Closures
    LoadDeref,          // Load from closure cell
    StoreDeref,         // Store to closure cell
    LoadClosure,        // Load closure cell for creating closure
    MakeClosure,        // Create a closure (function + closure cells)
}

/// Function flags for code objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionType {
    Regular,
    Generator,
    Coroutine,
    AsyncGenerator,
}

/// Compiled code object.
#[derive(Debug)]
pub struct CodeObject {
    pub name: String,
    pub bytecode: Vec<u8>,
    pub constants: Vec<Value>,
    pub names: Vec<String>,
    pub varnames: Vec<String>,
    pub freevars: Vec<String>,  // Variables accessed from enclosing scope
    pub cellvars: Vec<String>,  // Variables accessed by inner functions
    pub arg_count: u16,
    pub function_type: FunctionType,
}

impl CodeObject {
    pub fn new(name: String) -> Self {
        Self {
            name,
            bytecode: Vec::new(),
            constants: Vec::new(),
            names: Vec::new(),
            varnames: Vec::new(),
            freevars: Vec::new(),
            cellvars: Vec::new(),
            arg_count: 0,
            function_type: FunctionType::Regular,
        }
    }
}

/// Compiler for generating bytecode.
pub struct Compiler {
    code: CodeObject,
    locals: HashMap<String, u16>,
    scope_depth: usize,
    function_type: FunctionType,
    enclosing_vars: Vec<String>,  // Variables from enclosing scopes
    deref_vars: HashMap<String, u16>,  // Free variable name -> index
}

impl Compiler {
    /// Create a new compiler.
    pub fn new() -> Self {
        Self {
            code: CodeObject::new("<module>".into()),
            locals: HashMap::new(),
            scope_depth: 0,
            function_type: FunctionType::Regular,
            enclosing_vars: Vec::new(),
            deref_vars: HashMap::new(),
        }
    }

    /// Create a compiler for a function with specified type.
    pub fn new_function(name: String, func_type: FunctionType) -> Self {
        let mut code = CodeObject::new(name);
        code.function_type = func_type;
        Self {
            code,
            locals: HashMap::new(),
            scope_depth: 0,
            function_type: func_type,
            enclosing_vars: Vec::new(),
            deref_vars: HashMap::new(),
        }
    }

    /// Create a compiler for a nested function with enclosing scope info.
    pub fn new_nested_function(name: String, func_type: FunctionType, enclosing_vars: Vec<String>) -> Self {
        let mut code = CodeObject::new(name);
        code.function_type = func_type;
        Self {
            code,
            locals: HashMap::new(),
            scope_depth: 0,
            function_type: func_type,
            enclosing_vars,
            deref_vars: HashMap::new(),
        }
    }

    /// Compile a module.
    pub fn compile(mut self, module: &Module) -> Result<Rc<CodeObject>> {
        for stmt in &module.body {
            self.compile_stmt(stmt)?;
        }

        // Add implicit return None
        let none_idx = self.add_constant(Value::None);
        self.emit(OpCode::LoadConst);
        self.emit_u16(none_idx);
        self.emit(OpCode::Return);

        Ok(Rc::new(self.code))
    }

    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        match stmt {
            Stmt::Expr(expr) => {
                self.compile_expr(expr)?;
                self.emit(OpCode::Pop);
            }

            Stmt::Assign { targets, value } => {
                self.compile_expr(value)?;
                for target in targets {
                    self.compile_store(target)?;
                }
            }

            Stmt::AugAssign { target, op, value } => {
                self.compile_expr(target)?;
                self.compile_expr(value)?;
                self.emit_binary_op(*op);
                self.compile_store(target)?;
            }

            Stmt::Return(value) => {
                if let Some(expr) = value {
                    self.compile_expr(expr)?;
                } else {
                    let idx = self.add_constant(Value::None);
                    self.emit(OpCode::LoadConst);
                    self.emit_u16(idx);
                }
                self.emit(OpCode::Return);
            }

            Stmt::Pass => {
                self.emit(OpCode::Nop);
            }

            Stmt::Break => {
                // Placeholder - would need loop tracking
                return Err(Error::Compile("break outside loop".into()));
            }

            Stmt::Continue => {
                // Placeholder - would need loop tracking
                return Err(Error::Compile("continue outside loop".into()));
            }

            Stmt::If {
                test,
                body,
                elif_clauses,
                else_body,
            } => {
                self.compile_expr(test)?;
                let jump_to_else = self.emit_jump(OpCode::PopJumpIfFalse);

                for s in body {
                    self.compile_stmt(s)?;
                }
                let jump_to_end = self.emit_jump(OpCode::Jump);

                self.patch_jump(jump_to_else);

                // elif clauses
                let mut elif_end_jumps = Vec::new();
                for (elif_test, elif_body) in elif_clauses {
                    self.compile_expr(elif_test)?;
                    let elif_jump = self.emit_jump(OpCode::PopJumpIfFalse);

                    for s in elif_body {
                        self.compile_stmt(s)?;
                    }
                    elif_end_jumps.push(self.emit_jump(OpCode::Jump));
                    self.patch_jump(elif_jump);
                }

                // else
                for s in else_body {
                    self.compile_stmt(s)?;
                }

                self.patch_jump(jump_to_end);
                for jump in elif_end_jumps {
                    self.patch_jump(jump);
                }
            }

            Stmt::While { test, body } => {
                let loop_start = self.current_offset();

                self.compile_expr(test)?;
                let exit_jump = self.emit_jump(OpCode::PopJumpIfFalse);

                for s in body {
                    self.compile_stmt(s)?;
                }

                self.emit_loop(loop_start);
                self.patch_jump(exit_jump);
            }

            Stmt::For { target, iter, body } => {
                self.compile_expr(iter)?;
                self.emit(OpCode::GetIter);

                let loop_start = self.current_offset();
                let exit_jump = self.emit_jump(OpCode::ForIter);

                // Store loop variable - now safe since locals are stored separately from stack
                self.compile_store(&Expr::Identifier(target.clone()))?;

                for s in body {
                    self.compile_stmt(s)?;
                }

                self.emit_loop(loop_start);
                self.patch_jump(exit_jump);
                self.emit(OpCode::Pop); // Pop iterator
            }

            Stmt::FunctionDef { name, params, body, decorators } => {
                self.compile_function(name, params, body, FunctionType::Regular)?;
                // Function is now on stack
                // Apply decorators in reverse order (innermost first)
                for decorator in decorators.iter().rev() {
                    self.compile_expr(decorator)?;
                    // Swap function and decorator on stack, then call
                    self.emit(OpCode::RotTwo);
                    self.emit(OpCode::Call);
                    self.emit_u8(1); // 1 argument (the function)
                }
                // Store the (decorated) function in name
                let name_idx = self.add_name(name);
                self.emit(OpCode::StoreName);
                self.emit_u16(name_idx);
            }

            Stmt::AsyncFunctionDef { name, params, body, decorators } => {
                self.compile_function(name, params, body, FunctionType::Coroutine)?;
                // Function is now on stack
                // Apply decorators in reverse order (innermost first)
                for decorator in decorators.iter().rev() {
                    self.compile_expr(decorator)?;
                    // Swap function and decorator on stack, then call
                    self.emit(OpCode::RotTwo);
                    self.emit(OpCode::Call);
                    self.emit_u8(1); // 1 argument (the function)
                }
                // Store the (decorated) function in name
                let name_idx = self.add_name(name);
                self.emit(OpCode::StoreName);
                self.emit_u16(name_idx);
            }

            Stmt::AsyncFor { target, iter, body } => {
                self.compile_expr(iter)?;
                self.emit(OpCode::GetAiter);

                let loop_start = self.current_offset();
                let exit_jump = self.emit_jump(OpCode::GetAnext);

                // Store loop variable
                let idx = self.add_local(target);
                self.emit(OpCode::StoreFast);
                self.emit_u16(idx);

                for s in body {
                    self.compile_stmt(s)?;
                }

                self.emit_loop(loop_start);
                self.patch_jump(exit_jump);
                self.emit(OpCode::Pop); // Pop iterator
            }

            Stmt::With { items, body } | Stmt::AsyncWith { items, body } => {
                // Compile context managers
                for item in items {
                    self.compile_expr(&item.context_expr)?;
                    let exit_jump = self.emit_jump(OpCode::SetupWith);

                    // Store the __enter__ result if there's an 'as' clause
                    if let Some(ref var) = item.optional_vars {
                        let idx = self.add_local(var);
                        self.emit(OpCode::StoreFast);
                        self.emit_u16(idx);
                    } else {
                        self.emit(OpCode::Pop);
                    }

                    // Compile the body
                    for s in body {
                        self.compile_stmt(s)?;
                    }

                    // Cleanup
                    self.patch_jump(exit_jump);
                    self.emit(OpCode::WithCleanup);
                    break; // Only handle first item for simplicity
                }
            }

            Stmt::ClassDef { name, bases, body, decorators } => {
                // Load bases
                for base in bases {
                    self.compile_expr(base)?;
                }
                self.emit(OpCode::BuildTuple);
                self.emit_u16(bases.len() as u16);

                // Compile class body
                // Use "<module>" so that compile_store uses StoreName for class-level variables
                // This allows class attributes to be stored in globals and extracted as methods
                let mut class_compiler = Compiler::new();
                // Keep name as "<module>" so class-level variables use StoreName

                for s in body {
                    class_compiler.compile_stmt(s)?;
                }

                let none_idx = class_compiler.add_constant(Value::None);
                class_compiler.emit(OpCode::LoadConst);
                class_compiler.emit_u16(none_idx);
                class_compiler.emit(OpCode::Return);

                // Add class code as constant
                let code = Rc::new(class_compiler.code);
                let code_idx = self.add_constant(Value::Function(Rc::new(
                    crate::value::Function {
                        name: name.clone(),
                        code,
                        defaults: Vec::new(),
                        closure: None,
                    },
                )));

                self.emit(OpCode::LoadConst);
                self.emit_u16(code_idx);

                // Class name
                let name_const = self.add_constant(Value::String(Rc::new(name.clone())));
                self.emit(OpCode::LoadConst);
                self.emit_u16(name_const);

                self.emit(OpCode::BuildClass);

                // Apply decorators in reverse order (innermost first)
                for decorator in decorators.iter().rev() {
                    self.compile_expr(decorator)?;
                    // Swap class and decorator on stack, then call
                    self.emit(OpCode::RotTwo);
                    self.emit(OpCode::Call);
                    self.emit_u8(1); // 1 argument (the class)
                }

                // Store class
                let name_idx = self.add_name(name);
                self.emit(OpCode::StoreName);
                self.emit_u16(name_idx);
            }

            Stmt::Raise(value) => {
                if let Some(expr) = value {
                    self.compile_expr(expr)?;
                } else {
                    // Reraise current exception
                    self.emit(OpCode::Reraise);
                    return Ok(());
                }
                self.emit(OpCode::Raise);
            }

            Stmt::Try {
                body,
                handlers,
                else_body,
                finally_body,
            } => {
                // Compile try-except-else-finally
                // Structure:
                //   SETUP_EXCEPT handler_offset
                //   <try body>
                //   POP_EXCEPT
                //   <else body>
                //   JUMP end
                // handler:
                //   <except handlers>
                // end:
                //   <finally body>

                // Setup exception handler
                let except_jump = self.emit_jump(OpCode::SetupExcept);

                // Compile try body
                for s in body {
                    self.compile_stmt(s)?;
                }

                // Pop exception handler (no exception occurred)
                self.emit(OpCode::PopExcept);

                // Compile else body (runs if no exception)
                for s in else_body {
                    self.compile_stmt(s)?;
                }

                // Jump past handlers to finally
                let end_jump = self.emit_jump(OpCode::Jump);

                // Patch jump to handler
                self.patch_jump(except_jump);

                // Compile exception handlers
                let mut handler_end_jumps = Vec::new();
                for (i, handler) in handlers.iter().enumerate() {
                    // Store exception in variable if named
                    if let Some(ref name) = handler.name {
                        let idx = self.add_local(name);
                        self.emit(OpCode::StoreFast);
                        self.emit_u16(idx);
                    } else {
                        // Pop the exception value
                        self.emit(OpCode::Pop);
                    }

                    // Compile handler body
                    for s in &handler.body {
                        self.compile_stmt(s)?;
                    }

                    // Jump to finally/end after handler
                    if i < handlers.len() - 1 || !finally_body.is_empty() {
                        handler_end_jumps.push(self.emit_jump(OpCode::Jump));
                    }
                }

                // Patch end jump and handler end jumps
                self.patch_jump(end_jump);
                for jump in handler_end_jumps {
                    self.patch_jump(jump);
                }

                // Compile finally body
                for s in finally_body {
                    self.compile_stmt(s)?;
                }
                if !finally_body.is_empty() {
                    self.emit(OpCode::EndFinally);
                }
            }

            Stmt::Global(_) | Stmt::Nonlocal(_) => {
                // These are handled during name resolution
            }

            Stmt::Import(aliases) => {
                for alias in aliases {
                    // Simplified - just define the name
                    let name_idx = self.add_name(&alias.name);
                    let none_idx = self.add_constant(Value::None);
                    self.emit(OpCode::LoadConst);
                    self.emit_u16(none_idx);
                    self.emit(OpCode::StoreName);
                    self.emit_u16(name_idx);
                }
            }

            _ => {
                return Err(Error::Compile(format!(
                    "Unsupported statement: {:?}",
                    stmt
                )));
            }
        }

        Ok(())
    }

    fn compile_function(
        &mut self,
        name: &str,
        params: &[Param],
        body: &[Stmt],
        func_type: FunctionType,
    ) -> Result<()> {
        // Combine current locals with enclosing vars for nested scope
        let mut all_enclosing: Vec<String> = self.enclosing_vars.clone();
        all_enclosing.extend(self.get_all_locals());

        // Compile function body in new compiler with enclosing scope info
        let mut func_compiler = Compiler::new_nested_function(name.to_string(), func_type, all_enclosing);
        func_compiler.code.arg_count = params.len() as u16;

        // Add parameters as locals
        for param in params {
            func_compiler.add_local(&param.name);
        }

        // Compile body - check if it contains yield to detect generators
        let mut has_yield = false;
        for s in body {
            if Self::stmt_contains_yield(s) {
                has_yield = true;
                break;
            }
        }

        // Update function type if it contains yield
        if has_yield && func_type == FunctionType::Regular {
            func_compiler.code.function_type = FunctionType::Generator;
            func_compiler.function_type = FunctionType::Generator;
        } else if has_yield && func_type == FunctionType::Coroutine {
            func_compiler.code.function_type = FunctionType::AsyncGenerator;
            func_compiler.function_type = FunctionType::AsyncGenerator;
        }

        // Compile body
        for s in body {
            func_compiler.compile_stmt(s)?;
        }

        // Implicit return None
        let none_idx = func_compiler.add_constant(Value::None);
        func_compiler.emit(OpCode::LoadConst);
        func_compiler.emit_u16(none_idx);
        func_compiler.emit(OpCode::Return);

        // Check if function has free variables (needs closure)
        let freevars = func_compiler.code.freevars.clone();
        let has_closure = !freevars.is_empty();

        // Add function code as constant
        let code = Rc::new(func_compiler.code);
        let code_idx = self.add_constant(Value::Function(Rc::new(crate::value::Function {
            name: name.to_string(),
            code,
            defaults: Vec::new(),
            closure: None,  // Will be populated at runtime if needed
        })));

        self.emit(OpCode::LoadConst);
        self.emit_u16(code_idx);

        if has_closure {
            // Load closure cells for free variables
            // Each freevar needs to be captured from current scope
            for freevar in &freevars {
                // Load the value from current scope to create closure cell
                if let Some(&idx) = self.locals.get(freevar) {
                    self.emit(OpCode::LoadFast);
                    self.emit_u16(idx);
                } else if let Some(&idx) = self.deref_vars.get(freevar) {
                    // It's a free variable in our scope too - chain the closure
                    self.emit(OpCode::LoadDeref);
                    self.emit_u16(idx);
                } else if self.enclosing_vars.contains(&freevar.to_string()) {
                    // It's in an enclosing scope - we need to add it to our freevars too
                    let idx = self.add_freevar(freevar);
                    self.emit(OpCode::LoadDeref);
                    self.emit_u16(idx);
                } else {
                    // Try global
                    let idx = self.add_name(freevar);
                    self.emit(OpCode::LoadName);
                    self.emit_u16(idx);
                }
            }
            self.emit(OpCode::MakeClosure);
            self.emit_u16(freevars.len() as u16);
        } else {
            self.emit(OpCode::MakeFunction);
        }
        // Note: function is left on stack, caller should store it

        Ok(())
    }

    fn stmt_contains_yield(stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Expr(expr) => Self::expr_contains_yield(expr),
            Stmt::Assign { value, .. } => Self::expr_contains_yield(value),
            Stmt::AugAssign { value, .. } => Self::expr_contains_yield(value),
            Stmt::Return(Some(expr)) => Self::expr_contains_yield(expr),
            Stmt::If { test, body, elif_clauses, else_body } => {
                Self::expr_contains_yield(test)
                    || body.iter().any(|s| Self::stmt_contains_yield(s))
                    || elif_clauses.iter().any(|(e, b)| {
                        Self::expr_contains_yield(e) || b.iter().any(|s| Self::stmt_contains_yield(s))
                    })
                    || else_body.iter().any(|s| Self::stmt_contains_yield(s))
            }
            Stmt::While { test, body } => {
                Self::expr_contains_yield(test) || body.iter().any(|s| Self::stmt_contains_yield(s))
            }
            Stmt::For { iter, body, .. } => {
                Self::expr_contains_yield(iter) || body.iter().any(|s| Self::stmt_contains_yield(s))
            }
            _ => false,
        }
    }

    fn expr_contains_yield(expr: &Expr) -> bool {
        match expr {
            Expr::Yield(_) | Expr::YieldFrom(_) => true,
            Expr::BinaryOp { left, right, .. } => {
                Self::expr_contains_yield(left) || Self::expr_contains_yield(right)
            }
            Expr::UnaryOp { operand, .. } => Self::expr_contains_yield(operand),
            Expr::Call { func, args, .. } => {
                Self::expr_contains_yield(func) || args.iter().any(|a| Self::expr_contains_yield(a))
            }
            Expr::IfExpr { test, body, orelse } => {
                Self::expr_contains_yield(test)
                    || Self::expr_contains_yield(body)
                    || Self::expr_contains_yield(orelse)
            }
            _ => false,
        }
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<()> {
        match expr {
            Expr::Integer(n) => {
                let idx = self.add_constant(Value::Int(*n));
                self.emit(OpCode::LoadConst);
                self.emit_u16(idx);
            }

            Expr::Float(n) => {
                let idx = self.add_constant(Value::Float(*n));
                self.emit(OpCode::LoadConst);
                self.emit_u16(idx);
            }

            Expr::String(s) => {
                let idx = self.add_constant(Value::String(Rc::new(s.clone())));
                self.emit(OpCode::LoadConst);
                self.emit_u16(idx);
            }

            Expr::Bool(b) => {
                let idx = self.add_constant(Value::Bool(*b));
                self.emit(OpCode::LoadConst);
                self.emit_u16(idx);
            }

            Expr::None => {
                let idx = self.add_constant(Value::None);
                self.emit(OpCode::LoadConst);
                self.emit_u16(idx);
            }

            Expr::Identifier(name) => {
                if let Some(&idx) = self.locals.get(name) {
                    self.emit(OpCode::LoadFast);
                    self.emit_u16(idx);
                } else if let Some(&idx) = self.deref_vars.get(name) {
                    // Load from closure cell (free variable)
                    self.emit(OpCode::LoadDeref);
                    self.emit_u16(idx);
                } else if self.enclosing_vars.contains(&name.to_string()) {
                    // This is a free variable - add to freevars and use LoadDeref
                    let idx = self.add_freevar(name);
                    self.emit(OpCode::LoadDeref);
                    self.emit_u16(idx);
                } else {
                    let idx = self.add_name(name);
                    self.emit(OpCode::LoadName);
                    self.emit_u16(idx);
                }
            }

            Expr::BinaryOp { left, op, right } => {
                // Short-circuit for and/or
                if *op == BinaryOp::And {
                    self.compile_expr(left)?;
                    let jump = self.emit_jump(OpCode::JumpIfFalse);
                    self.emit(OpCode::Pop);
                    self.compile_expr(right)?;
                    self.patch_jump(jump);
                    return Ok(());
                }

                if *op == BinaryOp::Or {
                    self.compile_expr(left)?;
                    let jump = self.emit_jump(OpCode::JumpIfTrue);
                    self.emit(OpCode::Pop);
                    self.compile_expr(right)?;
                    self.patch_jump(jump);
                    return Ok(());
                }

                self.compile_expr(left)?;
                self.compile_expr(right)?;
                self.emit_binary_op(*op);
            }

            Expr::UnaryOp { op, operand } => {
                self.compile_expr(operand)?;
                match op {
                    UnaryOp::Neg => self.emit(OpCode::UnaryNeg),
                    UnaryOp::Not => self.emit(OpCode::UnaryNot),
                }
            }

            Expr::Compare { left, ops } => {
                self.compile_expr(left)?;
                for (op, right) in ops {
                    self.compile_expr(right)?;
                    match op {
                        CompareOp::Eq => self.emit(OpCode::CompareEq),
                        CompareOp::Ne => self.emit(OpCode::CompareNe),
                        CompareOp::Lt => self.emit(OpCode::CompareLt),
                        CompareOp::Le => self.emit(OpCode::CompareLe),
                        CompareOp::Gt => self.emit(OpCode::CompareGt),
                        CompareOp::Ge => self.emit(OpCode::CompareGe),
                        CompareOp::Is | CompareOp::IsNot => self.emit(OpCode::CompareIs),
                        CompareOp::In | CompareOp::NotIn => self.emit(OpCode::CompareIn),
                    }
                }
            }

            Expr::Call { func, args, .. } => {
                self.compile_expr(func)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                self.emit(OpCode::Call);
                self.emit_u8(args.len() as u8);
            }

            Expr::Attribute { value, attr } => {
                self.compile_expr(value)?;
                let idx = self.add_name(attr);
                self.emit(OpCode::LoadAttr);
                self.emit_u16(idx);
            }

            Expr::Subscript { value, index } => {
                self.compile_expr(value)?;
                self.compile_expr(index)?;
                self.emit(OpCode::LoadSubscript);
            }

            Expr::List(elements) => {
                for elem in elements {
                    self.compile_expr(elem)?;
                }
                self.emit(OpCode::BuildList);
                self.emit_u16(elements.len() as u16);
            }

            Expr::Dict(pairs) => {
                for (key, value) in pairs {
                    self.compile_expr(key)?;
                    self.compile_expr(value)?;
                }
                self.emit(OpCode::BuildDict);
                self.emit_u16(pairs.len() as u16);
            }

            Expr::Tuple(elements) => {
                for elem in elements {
                    self.compile_expr(elem)?;
                }
                self.emit(OpCode::BuildTuple);
                self.emit_u16(elements.len() as u16);
            }

            Expr::Lambda { params, body } => {
                let mut func_compiler = Compiler::new();
                func_compiler.code.name = "<lambda>".into();
                func_compiler.code.arg_count = params.len() as u16;

                for param in params {
                    func_compiler.add_local(&param.name);
                }

                func_compiler.compile_expr(body)?;
                func_compiler.emit(OpCode::Return);

                let code = Rc::new(func_compiler.code);
                let code_idx = self.add_constant(Value::Function(Rc::new(
                    crate::value::Function {
                        name: "<lambda>".into(),
                        code,
                        defaults: Vec::new(),
                        closure: None,
                    },
                )));

                self.emit(OpCode::LoadConst);
                self.emit_u16(code_idx);
                self.emit(OpCode::MakeFunction);
            }

            Expr::IfExpr { test, body, orelse } => {
                self.compile_expr(test)?;
                let else_jump = self.emit_jump(OpCode::PopJumpIfFalse);
                self.compile_expr(body)?;
                let end_jump = self.emit_jump(OpCode::Jump);
                self.patch_jump(else_jump);
                self.compile_expr(orelse)?;
                self.patch_jump(end_jump);
            }

            Expr::ListComp {
                element,
                target,
                iter,
                condition,
            } => {
                // Create empty list
                self.emit(OpCode::BuildList);
                self.emit_u16(0);

                // Get iterator
                self.compile_expr(iter)?;
                self.emit(OpCode::GetIter);

                let loop_start = self.current_offset();
                let exit_jump = self.emit_jump(OpCode::ForIter);

                // Store loop variable
                let idx = self.add_local(target);
                self.emit(OpCode::StoreFast);
                self.emit_u16(idx);

                // Condition
                let skip_append = if let Some(cond) = condition {
                    self.compile_expr(cond)?;
                    Some(self.emit_jump(OpCode::PopJumpIfFalse))
                } else {
                    None
                };

                // Append element to list
                self.emit(OpCode::Dup); // Dup the list
                self.compile_expr(element)?;
                self.emit(OpCode::ListAppend);

                if let Some(jump) = skip_append {
                    self.patch_jump(jump);
                }

                self.emit_loop(loop_start);
                self.patch_jump(exit_jump);
                self.emit(OpCode::Pop); // Pop iterator
            }

            // Generator expressions
            Expr::Yield(value) => {
                if let Some(expr) = value {
                    self.compile_expr(expr)?;
                } else {
                    let idx = self.add_constant(Value::None);
                    self.emit(OpCode::LoadConst);
                    self.emit_u16(idx);
                }
                self.emit(OpCode::YieldValue);
            }

            Expr::YieldFrom(expr) => {
                self.compile_expr(expr)?;
                self.emit(OpCode::GetIter);
                self.emit(OpCode::YieldFrom);
            }

            // Async expressions
            Expr::Await(expr) => {
                self.compile_expr(expr)?;
                self.emit(OpCode::GetAwaitable);
                // Load None for initial send value
                let idx = self.add_constant(Value::None);
                self.emit(OpCode::LoadConst);
                self.emit_u16(idx);
                self.emit(OpCode::YieldFrom);
            }
        }

        Ok(())
    }

    fn compile_store(&mut self, target: &Expr) -> Result<()> {
        match target {
            Expr::Identifier(name) => {
                if let Some(&idx) = self.locals.get(name) {
                    self.emit(OpCode::StoreFast);
                    self.emit_u16(idx);
                } else if self.code.name != "<module>" {
                    // Inside a function: create a new local variable
                    let idx = self.add_local(name);
                    self.emit(OpCode::StoreFast);
                    self.emit_u16(idx);
                } else {
                    // Module level: use global name
                    let idx = self.add_name(name);
                    self.emit(OpCode::StoreName);
                    self.emit_u16(idx);
                }
            }
            Expr::Attribute { value, attr } => {
                self.compile_expr(value)?;
                let idx = self.add_name(attr);
                self.emit(OpCode::StoreAttr);
                self.emit_u16(idx);
            }
            Expr::Subscript { value, index } => {
                self.compile_expr(value)?;
                self.compile_expr(index)?;
                self.emit(OpCode::StoreSubscript);
            }
            _ => return Err(Error::Compile("Invalid assignment target".into())),
        }
        Ok(())
    }

    fn emit_binary_op(&mut self, op: BinaryOp) {
        match op {
            BinaryOp::Add => self.emit(OpCode::BinaryAdd),
            BinaryOp::Sub => self.emit(OpCode::BinarySub),
            BinaryOp::Mul => self.emit(OpCode::BinaryMul),
            BinaryOp::Div => self.emit(OpCode::BinaryDiv),
            BinaryOp::FloorDiv => self.emit(OpCode::BinaryFloorDiv),
            BinaryOp::Mod => self.emit(OpCode::BinaryMod),
            BinaryOp::Pow => self.emit(OpCode::BinaryPow),
            BinaryOp::And | BinaryOp::Or => {} // Handled separately
        }
    }

    // Helper methods
    fn emit(&mut self, op: OpCode) {
        self.code.bytecode.push(op as u8);
    }

    fn emit_u8(&mut self, value: u8) {
        self.code.bytecode.push(value);
    }

    fn emit_u16(&mut self, value: u16) {
        self.code.bytecode.push((value >> 8) as u8);
        self.code.bytecode.push(value as u8);
    }

    fn emit_jump(&mut self, op: OpCode) -> usize {
        self.emit(op);
        let offset = self.code.bytecode.len();
        self.emit_u16(0); // Placeholder
        offset
    }

    fn patch_jump(&mut self, offset: usize) {
        let jump = self.code.bytecode.len() - offset - 2;
        self.code.bytecode[offset] = (jump >> 8) as u8;
        self.code.bytecode[offset + 1] = jump as u8;
    }

    fn emit_loop(&mut self, loop_start: usize) {
        self.emit(OpCode::Jump);
        let offset = self.code.bytecode.len() - loop_start + 2;
        let offset = -(offset as i16) as u16;
        self.emit_u16(offset);
    }

    fn current_offset(&self) -> usize {
        self.code.bytecode.len()
    }

    fn add_constant(&mut self, value: Value) -> u16 {
        self.code.constants.push(value);
        (self.code.constants.len() - 1) as u16
    }

    fn add_name(&mut self, name: &str) -> u16 {
        if let Some(idx) = self.code.names.iter().position(|n| n == name) {
            idx as u16
        } else {
            self.code.names.push(name.to_string());
            (self.code.names.len() - 1) as u16
        }
    }

    fn add_local(&mut self, name: &str) -> u16 {
        if let Some(&idx) = self.locals.get(name) {
            idx
        } else {
            let idx = self.code.varnames.len() as u16;
            self.code.varnames.push(name.to_string());
            self.locals.insert(name.to_string(), idx);
            idx
        }
    }

    fn add_freevar(&mut self, name: &str) -> u16 {
        if let Some(&idx) = self.deref_vars.get(name) {
            idx
        } else {
            let idx = self.code.freevars.len() as u16;
            self.code.freevars.push(name.to_string());
            self.deref_vars.insert(name.to_string(), idx);
            idx
        }
    }

    /// Get all local variable names (including parameters).
    fn get_all_locals(&self) -> Vec<String> {
        self.code.varnames.clone()
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}
