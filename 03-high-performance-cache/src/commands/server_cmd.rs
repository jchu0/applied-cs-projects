use crate::resp::RespValue;
use crate::storage::Database;

/// PING [message]
pub fn ping(args: &[RespValue]) -> RespValue {
    if args.is_empty() {
        RespValue::SimpleString("PONG".to_string())
    } else if let Some(msg) = args[0].as_str() {
        RespValue::bulk_string(msg)
    } else if let Some(data) = args[0].as_bytes() {
        RespValue::bulk(data.to_vec())
    } else {
        RespValue::SimpleString("PONG".to_string())
    }
}

/// ECHO message
pub fn echo(args: &[RespValue]) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'echo' command");
    }

    if let Some(data) = args[0].as_bytes() {
        RespValue::bulk(data.to_vec())
    } else {
        RespValue::null()
    }
}

/// INFO [section]
pub fn info(_args: &[RespValue], db: &mut Database) -> RespValue {
    let info = format!(
        "# Server\r\n\
        redis_version:0.1.0\r\n\
        redis_mode:standalone\r\n\
        \r\n\
        # Keyspace\r\n\
        db0:keys={},expires={}\r\n",
        db.len(),
        db.expires_count()
    );

    RespValue::bulk_string(info)
}

/// COMMAND
pub fn command(_args: &[RespValue]) -> RespValue {
    // Return list of supported commands
    let commands = vec![
        // String commands
        "GET", "SET", "SETNX", "SETEX", "PSETEX", "MGET", "MSET",
        "APPEND", "STRLEN", "INCR", "INCRBY", "DECR", "DECRBY", "GETSET",
        // Key commands
        "DEL", "EXISTS", "EXPIRE", "EXPIREAT", "PEXPIRE", "TTL", "PTTL",
        "PERSIST", "TYPE", "KEYS", "DBSIZE", "FLUSHDB", "RENAME", "RENAMENX",
        // Server commands
        "PING", "ECHO", "INFO", "COMMAND",
    ];

    let resp_commands: Vec<RespValue> = commands
        .into_iter()
        .map(|c| RespValue::bulk_string(c))
        .collect();

    RespValue::array(resp_commands)
}
