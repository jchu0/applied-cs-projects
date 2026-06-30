//! Lua scripting support for Redis-lite
//!
//! Implements EVAL, EVALSHA, SCRIPT commands for Lua script execution.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sha1::{Digest, Sha1};

/// Lua script execution engine
pub struct ScriptEngine {
    /// Cached scripts (SHA1 -> script source)
    scripts: Arc<Mutex<HashMap<String, String>>>,
    /// Script timeout in milliseconds
    timeout_ms: u64,
    /// Maximum script memory in bytes
    max_memory: usize,
}

/// Result of script execution
#[derive(Debug, Clone)]
pub enum ScriptResult {
    /// Nil value
    Nil,
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Boolean value
    Bool(bool),
    /// Array of values
    Array(Vec<ScriptResult>),
    /// Error message
    Error(String),
    /// Status (OK, QUEUED, etc.)
    Status(String),
}

/// Script execution context
pub struct ScriptContext {
    /// Keys accessed by the script
    pub keys: Vec<String>,
    /// Arguments passed to the script
    pub args: Vec<String>,
    /// Whether the script is read-only
    pub read_only: bool,
}

impl ScriptEngine {
    /// Create a new script engine
    pub fn new() -> Self {
        Self {
            scripts: Arc::new(Mutex::new(HashMap::new())),
            timeout_ms: 5000, // 5 second default timeout
            max_memory: 10 * 1024 * 1024, // 10MB default
        }
    }

    /// Create with custom configuration
    pub fn with_config(timeout_ms: u64, max_memory: usize) -> Self {
        Self {
            scripts: Arc::new(Mutex::new(HashMap::new())),
            timeout_ms,
            max_memory,
        }
    }

    /// Calculate SHA1 hash of script
    pub fn script_sha1(script: &str) -> String {
        let mut hasher = Sha1::new();
        hasher.update(script.as_bytes());
        let result = hasher.finalize();
        hex::encode(result)
    }

    /// Load a script into the cache
    pub fn load_script(&self, script: &str) -> String {
        let sha = Self::script_sha1(script);
        let mut scripts = self.scripts.lock().unwrap();
        scripts.insert(sha.clone(), script.to_string());
        sha
    }

    /// Check if a script exists
    pub fn script_exists(&self, sha: &str) -> bool {
        let scripts = self.scripts.lock().unwrap();
        scripts.contains_key(sha)
    }

    /// Get script by SHA1
    pub fn get_script(&self, sha: &str) -> Option<String> {
        let scripts = self.scripts.lock().unwrap();
        scripts.get(sha).cloned()
    }

    /// Flush all scripts
    pub fn flush_scripts(&self) {
        let mut scripts = self.scripts.lock().unwrap();
        scripts.clear();
    }

    /// Execute a script
    pub fn eval<F>(
        &self,
        script: &str,
        ctx: &ScriptContext,
        redis_call: F,
    ) -> Result<ScriptResult, String>
    where
        F: Fn(&str, &[String]) -> ScriptResult,
    {
        let start = Instant::now();
        let timeout = Duration::from_millis(self.timeout_ms);

        // Parse and execute the script
        let result = self.execute_script(script, ctx, &redis_call, start, timeout)?;

        Ok(result)
    }

    /// Execute script by SHA1
    pub fn evalsha<F>(
        &self,
        sha: &str,
        ctx: &ScriptContext,
        redis_call: F,
    ) -> Result<ScriptResult, String>
    where
        F: Fn(&str, &[String]) -> ScriptResult,
    {
        let script = self.get_script(sha)
            .ok_or_else(|| format!("NOSCRIPT No matching script. Use EVAL."))?;

        self.eval(&script, ctx, redis_call)
    }

    /// Internal script execution
    fn execute_script<F>(
        &self,
        script: &str,
        ctx: &ScriptContext,
        redis_call: &F,
        start: Instant,
        timeout: Duration,
    ) -> Result<ScriptResult, String>
    where
        F: Fn(&str, &[String]) -> ScriptResult,
    {
        // Simple script interpreter (basic Lua-like syntax)
        // For full Lua support, integrate with mlua or rlua crate

        let mut interpreter = ScriptInterpreter::new(ctx, redis_call);
        interpreter.set_timeout(start, timeout);
        interpreter.execute(script)
    }

    /// Kill a running script
    pub fn kill_script(&self) -> Result<(), String> {
        // In a real implementation, this would signal the script thread
        Ok(())
    }

    /// Get script debug info
    pub fn debug_info(&self) -> ScriptDebugInfo {
        let scripts = self.scripts.lock().unwrap();
        ScriptDebugInfo {
            cached_scripts: scripts.len(),
            timeout_ms: self.timeout_ms,
            max_memory: self.max_memory,
        }
    }
}

impl Default for ScriptEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Debug information for scripts
#[derive(Debug)]
pub struct ScriptDebugInfo {
    pub cached_scripts: usize,
    pub timeout_ms: u64,
    pub max_memory: usize,
}

/// Simple script interpreter
struct ScriptInterpreter<'a, F>
where
    F: Fn(&str, &[String]) -> ScriptResult,
{
    ctx: &'a ScriptContext,
    redis_call: &'a F,
    start: Option<Instant>,
    timeout: Option<Duration>,
    variables: HashMap<String, ScriptResult>,
}

impl<'a, F> ScriptInterpreter<'a, F>
where
    F: Fn(&str, &[String]) -> ScriptResult,
{
    fn new(ctx: &'a ScriptContext, redis_call: &'a F) -> Self {
        Self {
            ctx,
            redis_call,
            start: None,
            timeout: None,
            variables: HashMap::new(),
        }
    }

    fn set_timeout(&mut self, start: Instant, timeout: Duration) {
        self.start = Some(start);
        self.timeout = Some(timeout);
    }

    fn check_timeout(&self) -> Result<(), String> {
        if let (Some(start), Some(timeout)) = (self.start, self.timeout) {
            if start.elapsed() > timeout {
                return Err("BUSY script timeout".to_string());
            }
        }
        Ok(())
    }

    fn execute(&mut self, script: &str) -> Result<ScriptResult, String> {
        // Initialize KEYS and ARGV tables
        self.variables.insert(
            "KEYS".to_string(),
            ScriptResult::Array(
                self.ctx.keys.iter().map(|k| ScriptResult::String(k.clone())).collect()
            ),
        );
        self.variables.insert(
            "ARGV".to_string(),
            ScriptResult::Array(
                self.ctx.args.iter().map(|a| ScriptResult::String(a.clone())).collect()
            ),
        );

        // Parse and execute script lines
        let mut result = ScriptResult::Nil;

        for line in script.lines() {
            self.check_timeout()?;

            let line = line.trim();
            if line.is_empty() || line.starts_with("--") {
                continue;
            }

            result = self.execute_line(line)?;
        }

        Ok(result)
    }

    fn execute_line(&mut self, line: &str) -> Result<ScriptResult, String> {
        // Handle return statement
        if line.starts_with("return ") {
            let expr = &line[7..];
            return self.evaluate_expression(expr);
        }

        // Handle local variable assignment
        if line.starts_with("local ") {
            let rest = &line[6..];
            if let Some(eq_pos) = rest.find('=') {
                let var_name = rest[..eq_pos].trim().to_string();
                let expr = rest[eq_pos + 1..].trim();
                let value = self.evaluate_expression(expr)?;
                self.variables.insert(var_name, value);
                return Ok(ScriptResult::Nil);
            }
        }

        // Handle redis.call()
        if line.contains("redis.call(") || line.contains("redis.pcall(") {
            return self.execute_redis_call(line);
        }

        // Handle variable assignment
        if let Some(eq_pos) = line.find('=') {
            let var_name = line[..eq_pos].trim().to_string();
            let expr = line[eq_pos + 1..].trim();
            let value = self.evaluate_expression(expr)?;
            self.variables.insert(var_name, value);
            return Ok(ScriptResult::Nil);
        }

        Ok(ScriptResult::Nil)
    }

    fn evaluate_expression(&mut self, expr: &str) -> Result<ScriptResult, String> {
        let expr = expr.trim();

        // Handle nil
        if expr == "nil" {
            return Ok(ScriptResult::Nil);
        }

        // Handle boolean
        if expr == "true" {
            return Ok(ScriptResult::Bool(true));
        }
        if expr == "false" {
            return Ok(ScriptResult::Bool(false));
        }

        // Handle number
        if let Ok(n) = expr.parse::<i64>() {
            return Ok(ScriptResult::Integer(n));
        }

        // Handle string literal
        if (expr.starts_with('"') && expr.ends_with('"')) ||
           (expr.starts_with('\'') && expr.ends_with('\'')) {
            return Ok(ScriptResult::String(expr[1..expr.len()-1].to_string()));
        }

        // Handle table access (KEYS[1], ARGV[1])
        if let Some(result) = self.evaluate_table_access(expr) {
            return Ok(result);
        }

        // Handle redis.call
        if expr.contains("redis.call(") || expr.contains("redis.pcall(") {
            return self.execute_redis_call(expr);
        }

        // Handle variable reference
        if let Some(value) = self.variables.get(expr) {
            return Ok(value.clone());
        }

        Ok(ScriptResult::Nil)
    }

    fn evaluate_table_access(&self, expr: &str) -> Option<ScriptResult> {
        // Handle KEYS[n] or ARGV[n]
        let expr = expr.trim();

        for (table_name, prefix) in [("KEYS", "KEYS["), ("ARGV", "ARGV[")] {
            if expr.starts_with(prefix) && expr.ends_with(']') {
                let index_str = &expr[prefix.len()..expr.len()-1];
                if let Ok(index) = index_str.parse::<usize>() {
                    if let Some(ScriptResult::Array(arr)) = self.variables.get(table_name) {
                        // Lua arrays are 1-indexed
                        if index > 0 && index <= arr.len() {
                            return Some(arr[index - 1].clone());
                        }
                    }
                }
                return Some(ScriptResult::Nil);
            }
        }

        None
    }

    fn execute_redis_call(&mut self, line: &str) -> Result<ScriptResult, String> {
        // Extract call arguments
        let is_pcall = line.contains("redis.pcall(");

        let start = if is_pcall {
            line.find("redis.pcall(").map(|p| p + 12)
        } else {
            line.find("redis.call(").map(|p| p + 11)
        };

        let start = start.ok_or("Invalid redis call")?;
        let end = line[start..].find(')').ok_or("Missing closing parenthesis")?;
        let args_str = &line[start..start + end];

        // Parse arguments
        let args = self.parse_call_arguments(args_str)?;

        if args.is_empty() {
            return Err("redis.call requires at least one argument".to_string());
        }

        let cmd = args[0].clone();
        let cmd_args: Vec<String> = args[1..].to_vec();

        // Check read-only restriction
        if self.ctx.read_only && is_write_command(&cmd) {
            if is_pcall {
                return Ok(ScriptResult::Error(
                    "ERR Write commands not allowed in read-only scripts".to_string()
                ));
            } else {
                return Err("ERR Write commands not allowed in read-only scripts".to_string());
            }
        }

        // Execute the Redis command
        let result = (self.redis_call)(&cmd, &cmd_args);

        // For pcall, wrap errors
        if is_pcall {
            if let ScriptResult::Error(e) = &result {
                return Ok(ScriptResult::Error(e.clone()));
            }
        }

        Ok(result)
    }

    fn parse_call_arguments(&self, args_str: &str) -> Result<Vec<String>, String> {
        let mut args = Vec::new();
        let mut current = String::new();
        let mut in_string = false;
        let mut string_char = ' ';
        let mut chars = args_str.chars().peekable();

        while let Some(c) = chars.next() {
            if in_string {
                if c == string_char {
                    args.push(current.clone());
                    current.clear();
                    in_string = false;
                } else {
                    current.push(c);
                }
            } else {
                match c {
                    '"' | '\'' => {
                        in_string = true;
                        string_char = c;
                    }
                    ',' => {
                        if !current.trim().is_empty() {
                            // Evaluate expression for non-string arguments
                            let value = self.evaluate_arg(&current.trim())?;
                            args.push(value);
                        }
                        current.clear();
                    }
                    _ => {
                        current.push(c);
                    }
                }
            }
        }

        // Handle last argument
        if !current.trim().is_empty() {
            let value = self.evaluate_arg(&current.trim())?;
            args.push(value);
        }

        Ok(args)
    }

    fn evaluate_arg(&self, arg: &str) -> Result<String, String> {
        // Handle table access
        if let Some(result) = self.evaluate_table_access(arg) {
            return Ok(script_result_to_string(&result));
        }

        // Handle variable
        if let Some(value) = self.variables.get(arg) {
            return Ok(script_result_to_string(value));
        }

        // Return as-is
        Ok(arg.to_string())
    }
}

/// Check if command is a write command
fn is_write_command(cmd: &str) -> bool {
    let write_commands = [
        "SET", "SETNX", "SETEX", "PSETEX", "MSET", "MSETNX",
        "INCR", "INCRBY", "INCRBYFLOAT", "DECR", "DECRBY",
        "APPEND", "SETRANGE", "DEL", "UNLINK",
        "LPUSH", "LPUSHX", "RPUSH", "RPUSHX", "LPOP", "RPOP",
        "LSET", "LINSERT", "LREM", "LTRIM",
        "SADD", "SREM", "SPOP", "SMOVE",
        "HSET", "HSETNX", "HMSET", "HINCRBY", "HINCRBYFLOAT", "HDEL",
        "ZADD", "ZINCRBY", "ZREM", "ZREMRANGEBYRANK", "ZREMRANGEBYSCORE",
        "EXPIRE", "EXPIREAT", "PEXPIRE", "PEXPIREAT", "PERSIST",
        "RENAME", "RENAMENX", "COPY", "MOVE",
        "XADD", "XDEL", "XTRIM", "XSETID",
        "FLUSHDB", "FLUSHALL",
    ];

    write_commands.iter().any(|&c| c.eq_ignore_ascii_case(cmd))
}

/// Convert ScriptResult to string
fn script_result_to_string(result: &ScriptResult) -> String {
    match result {
        ScriptResult::Nil => String::new(),
        ScriptResult::String(s) => s.clone(),
        ScriptResult::Integer(n) => n.to_string(),
        ScriptResult::Bool(b) => if *b { "1".to_string() } else { "0".to_string() },
        ScriptResult::Status(s) => s.clone(),
        ScriptResult::Error(e) => e.clone(),
        ScriptResult::Array(_) => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_script_sha1() {
        let sha = ScriptEngine::script_sha1("return 1");
        assert_eq!(sha.len(), 40);

        // Same script should produce same hash
        let sha2 = ScriptEngine::script_sha1("return 1");
        assert_eq!(sha, sha2);

        // Different script should produce different hash
        let sha3 = ScriptEngine::script_sha1("return 2");
        assert_ne!(sha, sha3);
    }

    #[test]
    fn test_load_and_get_script() {
        let engine = ScriptEngine::new();

        let script = "return redis.call('GET', KEYS[1])";
        let sha = engine.load_script(script);

        assert!(engine.script_exists(&sha));
        assert_eq!(engine.get_script(&sha), Some(script.to_string()));
    }

    #[test]
    fn test_flush_scripts() {
        let engine = ScriptEngine::new();

        engine.load_script("return 1");
        engine.load_script("return 2");

        assert!(engine.debug_info().cached_scripts >= 2);

        engine.flush_scripts();

        assert_eq!(engine.debug_info().cached_scripts, 0);
    }

    #[test]
    fn test_simple_eval() {
        let engine = ScriptEngine::new();

        let ctx = ScriptContext {
            keys: vec!["key1".to_string()],
            args: vec!["arg1".to_string()],
            read_only: false,
        };

        let redis_call = |_cmd: &str, _args: &[String]| -> ScriptResult {
            ScriptResult::String("value".to_string())
        };

        let result = engine.eval("return 42", &ctx, redis_call).unwrap();

        match result {
            ScriptResult::Integer(n) => assert_eq!(n, 42),
            _ => panic!("Expected integer result"),
        }
    }

    #[test]
    fn test_keys_access() {
        let engine = ScriptEngine::new();

        let ctx = ScriptContext {
            keys: vec!["mykey".to_string()],
            args: vec![],
            read_only: false,
        };

        let redis_call = |_: &str, _: &[String]| ScriptResult::Nil;

        let result = engine.eval("return KEYS[1]", &ctx, redis_call).unwrap();

        match result {
            ScriptResult::String(s) => assert_eq!(s, "mykey"),
            _ => panic!("Expected string result"),
        }
    }

    #[test]
    fn test_redis_call() {
        let engine = ScriptEngine::new();

        let ctx = ScriptContext {
            keys: vec!["mykey".to_string()],
            args: vec![],
            read_only: false,
        };

        let redis_call = |cmd: &str, args: &[String]| -> ScriptResult {
            if cmd.eq_ignore_ascii_case("GET") && args.len() == 1 {
                ScriptResult::String("hello".to_string())
            } else {
                ScriptResult::Nil
            }
        };

        let script = "return redis.call('GET', KEYS[1])";
        let result = engine.eval(script, &ctx, redis_call).unwrap();

        match result {
            ScriptResult::String(s) => assert_eq!(s, "hello"),
            _ => panic!("Expected string result"),
        }
    }

    #[test]
    fn test_read_only_script() {
        let engine = ScriptEngine::new();

        let ctx = ScriptContext {
            keys: vec!["mykey".to_string()],
            args: vec!["value".to_string()],
            read_only: true,
        };

        let redis_call = |_: &str, _: &[String]| ScriptResult::Status("OK".to_string());

        let script = "return redis.call('SET', KEYS[1], ARGV[1])";
        let result = engine.eval(script, &ctx, redis_call);

        assert!(result.is_err());
    }

    #[test]
    fn test_is_write_command() {
        assert!(is_write_command("SET"));
        assert!(is_write_command("set"));
        assert!(is_write_command("DEL"));
        assert!(is_write_command("LPUSH"));

        assert!(!is_write_command("GET"));
        assert!(!is_write_command("LRANGE"));
        assert!(!is_write_command("SMEMBERS"));
    }
}
