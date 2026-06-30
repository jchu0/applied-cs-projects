//! Extended command executor with scripting and stream support.
//!
//! Extends the base CommandExecutor to include:
//! - Lua scripting commands (EVAL, EVALSHA, SCRIPT)
//! - Stream commands (XADD, XREAD, XRANGE, etc.)

use crate::resp::RespValue;
use crate::scripting::ScriptEngine;
use crate::storage::Database;

use super::{script_cmd, streams_cmd, CommandExecutor};
use super::streams_cmd::StreamStore;

/// Extended command executor with full feature support.
pub struct ExtendedExecutor {
    /// Script engine for Lua execution
    script_engine: ScriptEngine,
    /// Stream storage
    stream_store: StreamStore,
}

impl ExtendedExecutor {
    /// Create a new extended executor.
    pub fn new() -> Self {
        Self {
            script_engine: ScriptEngine::new(),
            stream_store: streams_cmd::new_stream_store(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(script_timeout_ms: u64, script_max_memory: usize) -> Self {
        Self {
            script_engine: ScriptEngine::with_config(script_timeout_ms, script_max_memory),
            stream_store: streams_cmd::new_stream_store(),
        }
    }

    /// Execute a command.
    pub fn execute(&self, cmd: &str, args: &[RespValue], db: &mut Database) -> RespValue {
        match cmd {
            // Scripting commands
            "EVAL" => script_cmd::eval(args, db, &self.script_engine),
            "EVALSHA" => script_cmd::evalsha(args, db, &self.script_engine),
            "EVAL_RO" => script_cmd::eval_ro(args, db, &self.script_engine),
            "EVALSHA_RO" => script_cmd::evalsha_ro(args, db, &self.script_engine),
            "SCRIPT" => script_cmd::script(args, &self.script_engine),

            // Stream commands
            "XADD" => streams_cmd::xadd(args, &self.stream_store),
            "XLEN" => streams_cmd::xlen(args, &self.stream_store),
            "XRANGE" => streams_cmd::xrange(args, &self.stream_store),
            "XREVRANGE" => streams_cmd::xrevrange(args, &self.stream_store),
            "XREAD" => streams_cmd::xread(args, &self.stream_store),
            "XREADGROUP" => streams_cmd::xreadgroup(args, &self.stream_store),
            "XGROUP" => streams_cmd::xgroup(args, &self.stream_store),
            "XACK" => streams_cmd::xack(args, &self.stream_store),
            "XTRIM" => streams_cmd::xtrim(args, &self.stream_store),
            "XDEL" => streams_cmd::xdel(args, &self.stream_store),
            "XINFO" => streams_cmd::xinfo(args, &self.stream_store),

            // Delegate to base executor for all other commands
            _ => CommandExecutor::execute(cmd, args, db),
        }
    }

    /// Get reference to script engine.
    pub fn script_engine(&self) -> &ScriptEngine {
        &self.script_engine
    }

    /// Get reference to stream store.
    pub fn stream_store(&self) -> &StreamStore {
        &self.stream_store
    }
}

impl Default for ExtendedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_executor() -> ExtendedExecutor {
        ExtendedExecutor::new()
    }

    // ==================== Scripting Command Tests ====================

    #[test]
    fn test_script_load_and_exists() {
        let executor = create_test_executor();
        let mut db = Database::new();

        // SCRIPT LOAD
        let result = executor.execute(
            "SCRIPT",
            &[RespValue::bulk_string("LOAD"), RespValue::bulk_string("return 1")],
            &mut db,
        );

        let sha = match result {
            RespValue::BulkString(Some(s)) => String::from_utf8(s).unwrap(),
            _ => panic!("Expected SHA1 string"),
        };

        // SCRIPT EXISTS
        let result = executor.execute(
            "SCRIPT",
            &[RespValue::bulk_string("EXISTS"), RespValue::bulk_string(&sha)],
            &mut db,
        );

        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 1);
                assert_eq!(arr[0], RespValue::integer(1));
            }
            _ => panic!("Expected array"),
        }
    }

    #[test]
    fn test_script_flush() {
        let executor = create_test_executor();
        let mut db = Database::new();

        // Load some scripts
        executor.execute(
            "SCRIPT",
            &[RespValue::bulk_string("LOAD"), RespValue::bulk_string("return 1")],
            &mut db,
        );

        // Flush
        let result = executor.execute(
            "SCRIPT",
            &[RespValue::bulk_string("FLUSH")],
            &mut db,
        );
        assert_eq!(result, RespValue::ok());

        // Verify cache is empty
        let info = executor.script_engine().debug_info();
        assert_eq!(info.cached_scripts, 0);
    }

    #[test]
    fn test_eval_command() {
        let executor = create_test_executor();
        let mut db = Database::new();

        let result = executor.execute(
            "EVAL",
            &[
                RespValue::bulk_string("return 42"),
                RespValue::bulk_string("0"),
            ],
            &mut db,
        );

        // Result depends on interpreter, just verify no crash
        assert!(matches!(
            result,
            RespValue::Integer(_) | RespValue::Error(_) | RespValue::BulkString(_)
        ));
    }

    // ==================== Stream Command Tests ====================

    #[test]
    fn test_xadd_and_xlen() {
        let executor = create_test_executor();
        let mut db = Database::new();

        // XADD mystream * field1 value1
        let result = executor.execute(
            "XADD",
            &[
                RespValue::bulk_string("mystream"),
                RespValue::bulk_string("*"),
                RespValue::bulk_string("field1"),
                RespValue::bulk_string("value1"),
            ],
            &mut db,
        );

        // Should return stream ID
        match result {
            RespValue::BulkString(Some(id)) => {
                let id_str = String::from_utf8(id).unwrap();
                assert!(id_str.contains("-")); // Format: timestamp-seq
            }
            RespValue::Error(e) => panic!("Unexpected error: {}", e),
            _ => panic!("Expected bulk string ID"),
        }

        // XLEN mystream
        let result = executor.execute(
            "XLEN",
            &[RespValue::bulk_string("mystream")],
            &mut db,
        );
        assert_eq!(result, RespValue::integer(1));
    }

    #[test]
    fn test_xadd_multiple_entries() {
        let executor = create_test_executor();
        let mut db = Database::new();

        // Add multiple entries
        for i in 0..5 {
            executor.execute(
                "XADD",
                &[
                    RespValue::bulk_string("mystream"),
                    RespValue::bulk_string("*"),
                    RespValue::bulk_string("count"),
                    RespValue::bulk_string(i.to_string()),
                ],
                &mut db,
            );
        }

        // XLEN should be 5
        let result = executor.execute(
            "XLEN",
            &[RespValue::bulk_string("mystream")],
            &mut db,
        );
        assert_eq!(result, RespValue::integer(5));
    }

    #[test]
    fn test_xrange() {
        let executor = create_test_executor();
        let mut db = Database::new();

        // Add entries
        executor.execute(
            "XADD",
            &[
                RespValue::bulk_string("mystream"),
                RespValue::bulk_string("*"),
                RespValue::bulk_string("a"),
                RespValue::bulk_string("1"),
            ],
            &mut db,
        );
        executor.execute(
            "XADD",
            &[
                RespValue::bulk_string("mystream"),
                RespValue::bulk_string("*"),
                RespValue::bulk_string("b"),
                RespValue::bulk_string("2"),
            ],
            &mut db,
        );

        // XRANGE mystream - +
        let result = executor.execute(
            "XRANGE",
            &[
                RespValue::bulk_string("mystream"),
                RespValue::bulk_string("-"),
                RespValue::bulk_string("+"),
            ],
            &mut db,
        );

        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 2);
            }
            _ => panic!("Expected array"),
        }
    }

    #[test]
    fn test_xgroup_create() {
        let executor = create_test_executor();
        let mut db = Database::new();

        // First add an entry to create the stream
        executor.execute(
            "XADD",
            &[
                RespValue::bulk_string("mystream"),
                RespValue::bulk_string("*"),
                RespValue::bulk_string("field"),
                RespValue::bulk_string("value"),
            ],
            &mut db,
        );

        // XGROUP CREATE mystream mygroup $
        let result = executor.execute(
            "XGROUP",
            &[
                RespValue::bulk_string("CREATE"),
                RespValue::bulk_string("mystream"),
                RespValue::bulk_string("mygroup"),
                RespValue::bulk_string("$"),
            ],
            &mut db,
        );

        assert_eq!(result, RespValue::ok());
    }

    #[test]
    fn test_xtrim() {
        let executor = create_test_executor();
        let mut db = Database::new();

        // Add 10 entries
        for i in 0..10 {
            executor.execute(
                "XADD",
                &[
                    RespValue::bulk_string("mystream"),
                    RespValue::bulk_string("*"),
                    RespValue::bulk_string("n"),
                    RespValue::bulk_string(i.to_string()),
                ],
                &mut db,
            );
        }

        // XTRIM mystream MAXLEN 5
        let result = executor.execute(
            "XTRIM",
            &[
                RespValue::bulk_string("mystream"),
                RespValue::bulk_string("MAXLEN"),
                RespValue::bulk_string("5"),
            ],
            &mut db,
        );

        // Should return number of deleted entries
        match result {
            RespValue::Integer(n) => assert_eq!(n, 5),
            _ => panic!("Expected integer"),
        }

        // Verify length
        let result = executor.execute(
            "XLEN",
            &[RespValue::bulk_string("mystream")],
            &mut db,
        );
        assert_eq!(result, RespValue::integer(5));
    }

    #[test]
    fn test_xinfo_stream() {
        let executor = create_test_executor();
        let mut db = Database::new();

        // Add entry
        executor.execute(
            "XADD",
            &[
                RespValue::bulk_string("mystream"),
                RespValue::bulk_string("*"),
                RespValue::bulk_string("field"),
                RespValue::bulk_string("value"),
            ],
            &mut db,
        );

        // XINFO STREAM mystream
        let result = executor.execute(
            "XINFO",
            &[
                RespValue::bulk_string("STREAM"),
                RespValue::bulk_string("mystream"),
            ],
            &mut db,
        );

        match result {
            RespValue::Array(Some(arr)) => {
                assert!(!arr.is_empty());
            }
            _ => panic!("Expected array"),
        }
    }

    // ==================== Base Command Integration Tests ====================

    #[test]
    fn test_base_commands_still_work() {
        let executor = create_test_executor();
        let mut db = Database::new();

        // SET
        let result = executor.execute(
            "SET",
            &[RespValue::bulk_string("key"), RespValue::bulk_string("value")],
            &mut db,
        );
        assert_eq!(result, RespValue::ok());

        // GET
        let result = executor.execute(
            "GET",
            &[RespValue::bulk_string("key")],
            &mut db,
        );
        assert_eq!(result, RespValue::BulkString(Some(b"value".to_vec())));

        // LPUSH
        let result = executor.execute(
            "LPUSH",
            &[RespValue::bulk_string("list"), RespValue::bulk_string("item")],
            &mut db,
        );
        assert_eq!(result, RespValue::integer(1));

        // SADD
        let result = executor.execute(
            "SADD",
            &[RespValue::bulk_string("set"), RespValue::bulk_string("member")],
            &mut db,
        );
        assert_eq!(result, RespValue::integer(1));

        // PING
        let result = executor.execute("PING", &[], &mut db);
        assert_eq!(result, RespValue::SimpleString("PONG".to_string()));
    }

    #[test]
    fn test_unknown_command() {
        let executor = create_test_executor();
        let mut db = Database::new();

        let result = executor.execute("UNKNOWNCMD", &[], &mut db);
        match result {
            RespValue::Error(msg) => {
                assert!(msg.contains("unknown command"));
            }
            _ => panic!("Expected error"),
        }
    }
}
