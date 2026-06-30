//! Transaction command implementations
//!
//! Note: Full transaction support requires client state management at the server level.
//! These implementations provide command parsing and response formatting.

use crate::resp::RespValue;
use crate::storage::Database;

/// MULTI - Start a transaction
/// Returns OK
pub fn multi(_args: &[RespValue], _db: &mut Database) -> RespValue {
    // Note: Actual MULTI handling requires client transaction state
    // This is typically managed at the server level
    RespValue::ok()
}

/// EXEC - Execute queued commands
/// Returns array of command results
pub fn exec(_args: &[RespValue], _db: &mut Database) -> RespValue {
    // Note: Actual EXEC requires executing queued commands
    // This is managed at the server level
    RespValue::error("ERR EXEC without MULTI")
}

/// DISCARD - Discard queued commands
/// Returns OK
pub fn discard(_args: &[RespValue], _db: &mut Database) -> RespValue {
    // Note: Actual DISCARD requires clearing transaction state
    // This is managed at the server level
    RespValue::error("ERR DISCARD without MULTI")
}

/// WATCH key [key ...] - Watch keys for optimistic locking
/// Returns OK
pub fn watch(args: &[RespValue], _db: &mut Database) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'watch' command");
    }

    // Note: Actual WATCH requires recording key versions
    // This is managed at the server level
    RespValue::ok()
}

/// UNWATCH - Unwatch all keys
/// Returns OK
pub fn unwatch(_args: &[RespValue], _db: &mut Database) -> RespValue {
    // Note: Actual UNWATCH requires clearing watched keys
    // This is managed at the server level
    RespValue::ok()
}
