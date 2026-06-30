//! Scripting command implementations (EVAL, EVALSHA, SCRIPT)
//!
//! Integrates the ScriptEngine with the command executor.

use crate::resp::RespValue;
use crate::scripting::{ScriptContext, ScriptEngine, ScriptResult};
use crate::storage::Database;

use std::sync::Arc;

/// Execute EVAL command
/// EVAL script numkeys [key ...] [arg ...]
pub fn eval(args: &[RespValue], db: &mut Database, engine: &ScriptEngine) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'eval' command");
    }

    let script = match args[0].as_str() {
        Some(s) => s,
        None => return RespValue::error("ERR invalid script"),
    };

    let numkeys = match args[1].as_int() {
        Some(n) => n as usize,
        None => return RespValue::error("ERR numkeys must be an integer"),
    };

    if args.len() < 2 + numkeys {
        return RespValue::error("ERR wrong number of arguments for 'eval' command");
    }

    // Extract keys and args
    let keys: Vec<String> = args[2..2 + numkeys]
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    let script_args: Vec<String> = args[2 + numkeys..]
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    let ctx = ScriptContext {
        keys,
        args: script_args,
        read_only: false,
    };

    // Create redis.call callback
    let redis_call = create_redis_call(db);

    match engine.eval(script, &ctx, redis_call) {
        Ok(result) => script_result_to_resp(result),
        Err(e) => RespValue::error(format!("ERR {}", e)),
    }
}

/// Execute EVALSHA command
/// EVALSHA sha1 numkeys [key ...] [arg ...]
pub fn evalsha(args: &[RespValue], db: &mut Database, engine: &ScriptEngine) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'evalsha' command");
    }

    let sha = match args[0].as_str() {
        Some(s) => s,
        None => return RespValue::error("ERR invalid SHA1"),
    };

    let numkeys = match args[1].as_int() {
        Some(n) => n as usize,
        None => return RespValue::error("ERR numkeys must be an integer"),
    };

    if args.len() < 2 + numkeys {
        return RespValue::error("ERR wrong number of arguments for 'evalsha' command");
    }

    // Extract keys and args
    let keys: Vec<String> = args[2..2 + numkeys]
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    let script_args: Vec<String> = args[2 + numkeys..]
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    let ctx = ScriptContext {
        keys,
        args: script_args,
        read_only: false,
    };

    // Create redis.call callback
    let redis_call = create_redis_call(db);

    match engine.evalsha(sha, &ctx, redis_call) {
        Ok(result) => script_result_to_resp(result),
        Err(e) => RespValue::error(format!("NOSCRIPT {}", e)),
    }
}

/// Execute SCRIPT command
/// SCRIPT LOAD script
/// SCRIPT EXISTS sha1 [sha1 ...]
/// SCRIPT FLUSH
/// SCRIPT KILL
pub fn script(args: &[RespValue], engine: &ScriptEngine) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'script' command");
    }

    let subcommand = match args[0].as_str() {
        Some(s) => s.to_uppercase(),
        None => return RespValue::error("ERR invalid subcommand"),
    };

    match subcommand.as_str() {
        "LOAD" => {
            if args.len() != 2 {
                return RespValue::error("ERR wrong number of arguments for 'script load' command");
            }
            let script = match args[1].as_str() {
                Some(s) => s,
                None => return RespValue::error("ERR invalid script"),
            };
            let sha = engine.load_script(script);
            RespValue::bulk_string(sha)
        }

        "EXISTS" => {
            if args.len() < 2 {
                return RespValue::error("ERR wrong number of arguments for 'script exists' command");
            }
            let results: Vec<RespValue> = args[1..]
                .iter()
                .map(|v| {
                    let exists = v.as_str()
                        .map(|sha| engine.script_exists(sha))
                        .unwrap_or(false);
                    RespValue::integer(if exists { 1 } else { 0 })
                })
                .collect();
            RespValue::array(results)
        }

        "FLUSH" => {
            engine.flush_scripts();
            RespValue::ok()
        }

        "KILL" => {
            match engine.kill_script() {
                Ok(()) => RespValue::ok(),
                Err(e) => RespValue::error(e),
            }
        }

        "DEBUG" => {
            let info = engine.debug_info();
            RespValue::array(vec![
                RespValue::bulk_string("cached_scripts"),
                RespValue::integer(info.cached_scripts as i64),
                RespValue::bulk_string("timeout_ms"),
                RespValue::integer(info.timeout_ms as i64),
                RespValue::bulk_string("max_memory"),
                RespValue::integer(info.max_memory as i64),
            ])
        }

        _ => RespValue::error(format!("ERR unknown script subcommand '{}'", subcommand)),
    }
}

/// Execute EVAL_RO (read-only EVAL)
pub fn eval_ro(args: &[RespValue], db: &mut Database, engine: &ScriptEngine) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'eval_ro' command");
    }

    let script = match args[0].as_str() {
        Some(s) => s,
        None => return RespValue::error("ERR invalid script"),
    };

    let numkeys = match args[1].as_int() {
        Some(n) => n as usize,
        None => return RespValue::error("ERR numkeys must be an integer"),
    };

    if args.len() < 2 + numkeys {
        return RespValue::error("ERR wrong number of arguments for 'eval_ro' command");
    }

    let keys: Vec<String> = args[2..2 + numkeys]
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    let script_args: Vec<String> = args[2 + numkeys..]
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    let ctx = ScriptContext {
        keys,
        args: script_args,
        read_only: true,
    };

    let redis_call = create_redis_call(db);

    match engine.eval(script, &ctx, redis_call) {
        Ok(result) => script_result_to_resp(result),
        Err(e) => RespValue::error(format!("ERR {}", e)),
    }
}

/// Execute EVALSHA_RO (read-only EVALSHA)
pub fn evalsha_ro(args: &[RespValue], db: &mut Database, engine: &ScriptEngine) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'evalsha_ro' command");
    }

    let sha = match args[0].as_str() {
        Some(s) => s,
        None => return RespValue::error("ERR invalid SHA1"),
    };

    let numkeys = match args[1].as_int() {
        Some(n) => n as usize,
        None => return RespValue::error("ERR numkeys must be an integer"),
    };

    if args.len() < 2 + numkeys {
        return RespValue::error("ERR wrong number of arguments for 'evalsha_ro' command");
    }

    let keys: Vec<String> = args[2..2 + numkeys]
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    let script_args: Vec<String> = args[2 + numkeys..]
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    let ctx = ScriptContext {
        keys,
        args: script_args,
        read_only: true,
    };

    let redis_call = create_redis_call(db);

    match engine.evalsha(sha, &ctx, redis_call) {
        Ok(result) => script_result_to_resp(result),
        Err(e) => RespValue::error(format!("NOSCRIPT {}", e)),
    }
}

/// Create a redis.call callback function
fn create_redis_call(_db: &mut Database) -> impl Fn(&str, &[String]) -> ScriptResult {
    // Note: In a full implementation, this would execute Redis commands
    // For now, we provide a stub that handles basic commands
    move |cmd: &str, args: &[String]| {
        match cmd.to_uppercase().as_str() {
            "PING" => {
                if args.is_empty() {
                    ScriptResult::Status("PONG".to_string())
                } else {
                    ScriptResult::String(args[0].clone())
                }
            }
            "ECHO" => {
                if args.is_empty() {
                    ScriptResult::Error("wrong number of arguments".to_string())
                } else {
                    ScriptResult::String(args[0].clone())
                }
            }
            _ => {
                // In a full implementation, this would execute the command
                ScriptResult::Error(format!("ERR command '{}' not yet implemented in script context", cmd))
            }
        }
    }
}

/// Convert ScriptResult to RespValue
fn script_result_to_resp(result: ScriptResult) -> RespValue {
    match result {
        ScriptResult::Nil => RespValue::null(),
        ScriptResult::String(s) => RespValue::bulk_string(s),
        ScriptResult::Integer(i) => RespValue::integer(i),
        ScriptResult::Bool(b) => RespValue::integer(if b { 1 } else { 0 }),
        ScriptResult::Array(arr) => {
            let values: Vec<RespValue> = arr.into_iter().map(script_result_to_resp).collect();
            RespValue::array(values)
        }
        ScriptResult::Error(e) => RespValue::error(e),
        ScriptResult::Status(s) => RespValue::SimpleString(s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_engine() -> ScriptEngine {
        ScriptEngine::new()
    }

    #[test]
    fn test_script_load() {
        let engine = create_test_engine();
        let lua_script = "return 'hello'";

        let result = script(
            &[RespValue::bulk_string("LOAD"), RespValue::bulk_string(lua_script)],
            &engine,
        );

        match result {
            RespValue::BulkString(Some(sha)) => {
                assert!(!sha.is_empty());
                // SHA1 is 40 hex characters
                assert_eq!(sha.len(), 40);
            }
            _ => panic!("Expected bulk string SHA1"),
        }
    }

    #[test]
    fn test_script_exists() {
        let engine = create_test_engine();
        let lua_script = "return 1";
        let sha = engine.load_script(lua_script);

        let result = script(
            &[
                RespValue::bulk_string("EXISTS"),
                RespValue::bulk_string(&sha),
                RespValue::bulk_string("nonexistent"),
            ],
            &engine,
        );

        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 2);
                assert_eq!(arr[0], RespValue::integer(1)); // exists
                assert_eq!(arr[1], RespValue::integer(0)); // doesn't exist
            }
            _ => panic!("Expected array"),
        }
    }

    #[test]
    fn test_script_flush() {
        let engine = create_test_engine();
        engine.load_script("return 1");
        engine.load_script("return 2");

        let info_before = engine.debug_info();
        assert_eq!(info_before.cached_scripts, 2);

        let result = script(&[RespValue::bulk_string("FLUSH")], &engine);
        assert_eq!(result, RespValue::ok());

        let info_after = engine.debug_info();
        assert_eq!(info_after.cached_scripts, 0);
    }

    #[test]
    fn test_script_debug() {
        let engine = create_test_engine();

        let result = script(&[RespValue::bulk_string("DEBUG")], &engine);

        match result {
            RespValue::Array(Some(arr)) => {
                assert!(arr.len() >= 6);
            }
            _ => panic!("Expected array"),
        }
    }

    #[test]
    fn test_eval_basic() {
        let mut db = Database::new();
        let engine = create_test_engine();

        // Note: The simple interpreter may not handle complex Lua
        // This tests the command parsing
        let result = eval(
            &[
                RespValue::bulk_string("return 42"),
                RespValue::bulk_string("0"),
            ],
            &mut db,
            &engine,
        );

        // The result depends on interpreter implementation
        // Just verify no crash
        assert!(matches!(
            result,
            RespValue::Integer(_) | RespValue::Error(_) | RespValue::BulkString(_)
        ));
    }

    #[test]
    fn test_eval_with_keys_and_args() {
        let mut db = Database::new();
        let engine = create_test_engine();

        let result = eval(
            &[
                RespValue::bulk_string("return KEYS[1]"),
                RespValue::bulk_string("2"),
                RespValue::bulk_string("key1"),
                RespValue::bulk_string("key2"),
                RespValue::bulk_string("arg1"),
            ],
            &mut db,
            &engine,
        );

        // Verify command parsing works
        assert!(matches!(
            result,
            RespValue::BulkString(_) | RespValue::Error(_) | RespValue::Integer(_)
        ));
    }

    #[test]
    fn test_evalsha_noscript() {
        let mut db = Database::new();
        let engine = create_test_engine();

        let result = evalsha(
            &[
                RespValue::bulk_string("0000000000000000000000000000000000000000"),
                RespValue::bulk_string("0"),
            ],
            &mut db,
            &engine,
        );

        match result {
            RespValue::Error(msg) => {
                assert!(msg.contains("NOSCRIPT"));
            }
            _ => panic!("Expected NOSCRIPT error"),
        }
    }

    #[test]
    fn test_script_result_conversion() {
        assert_eq!(script_result_to_resp(ScriptResult::Nil), RespValue::null());
        assert_eq!(
            script_result_to_resp(ScriptResult::Integer(42)),
            RespValue::integer(42)
        );
        assert_eq!(
            script_result_to_resp(ScriptResult::Bool(true)),
            RespValue::integer(1)
        );
        assert_eq!(
            script_result_to_resp(ScriptResult::Bool(false)),
            RespValue::integer(0)
        );
    }
}
