//! Master-specific replication state and operations

use std::io::{self, Write};
use std::sync::Arc;

use crate::storage::Database;
use crate::persistence::RDB;

/// Master replication state
pub struct MasterState {
    /// RDB handler for full sync
    rdb: Arc<RDB>,
}

impl MasterState {
    /// Create a new master state
    pub fn new(rdb_path: String) -> Self {
        Self {
            rdb: Arc::new(RDB::new(rdb_path)),
        }
    }

    /// Generate RDB snapshot for full synchronization
    pub fn generate_rdb(&self, db: &Database) -> io::Result<Vec<u8>> {
        // Create temporary file for RDB
        let temp_path = format!("/tmp/redis-lite-sync-{}.rdb", std::process::id());

        // Create temporary RDB handler
        let temp_rdb = RDB::new(temp_path.clone());
        temp_rdb.save(db)?;

        // Read the file contents
        let data = std::fs::read(&temp_path)?;

        // Clean up
        let _ = std::fs::remove_file(&temp_path);

        Ok(data)
    }

    /// Format PSYNC response for full sync
    pub fn fullresync_response(&self, repl_id: &str, offset: u64) -> String {
        format!("+FULLRESYNC {} {}\r\n", repl_id, offset)
    }

    /// Format PSYNC response for partial sync
    pub fn continue_response(&self, repl_id: &str, offset: u64) -> String {
        format!("+CONTINUE {} {}\r\n", repl_id, offset)
    }

    /// Format replication command for propagation
    pub fn format_command(cmd: &str, args: &[&[u8]]) -> Vec<u8> {
        let mut result = Vec::new();

        // RESP array format
        result.extend_from_slice(format!("*{}\r\n", args.len() + 1).as_bytes());

        // Command
        result.extend_from_slice(format!("${}\r\n", cmd.len()).as_bytes());
        result.extend_from_slice(cmd.as_bytes());
        result.extend_from_slice(b"\r\n");

        // Arguments
        for arg in args {
            result.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
            result.extend_from_slice(arg);
            result.extend_from_slice(b"\r\n");
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_command() {
        let cmd = MasterState::format_command("SET", &[b"key", b"value"]);
        let expected = b"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n";
        assert_eq!(cmd, expected);
    }

    #[test]
    fn test_fullresync_response() {
        let master = MasterState::new("/tmp/test.rdb".to_string());
        let response = master.fullresync_response("abc123", 0);
        assert!(response.starts_with("+FULLRESYNC abc123 0"));
    }
}
