use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::resp::RespValue;
use crate::storage::Database;

/// DEL key [key ...]
pub fn del(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'del' command");
    }

    let mut deleted = 0;
    for arg in args {
        if let Some(key) = arg.as_str() {
            if db.delete(key) {
                deleted += 1;
            }
        }
    }

    RespValue::integer(deleted)
}

/// EXISTS key [key ...]
pub fn exists(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'exists' command");
    }

    let mut count = 0;
    for arg in args {
        if let Some(key) = arg.as_str() {
            if db.exists(key) {
                count += 1;
            }
        }
    }

    RespValue::integer(count)
}

/// EXPIRE key seconds
pub fn expire(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'expire' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let seconds = match args[1].as_int() {
        Some(s) if s >= 0 => s as u64,
        _ => return RespValue::error("ERR invalid expire time"),
    };

    if db.expire(key, Duration::from_secs(seconds)) {
        RespValue::integer(1)
    } else {
        RespValue::integer(0)
    }
}

/// EXPIREAT key timestamp
pub fn expireat(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'expireat' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let timestamp = match args[1].as_int() {
        Some(t) if t >= 0 => t as u64,
        _ => return RespValue::error("ERR invalid expire time"),
    };

    // Convert Unix timestamp to Instant
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if timestamp <= now {
        // Already expired
        db.delete(key);
        RespValue::integer(1)
    } else {
        let duration = Duration::from_secs(timestamp - now);
        if db.expire(key, duration) {
            RespValue::integer(1)
        } else {
            RespValue::integer(0)
        }
    }
}

/// PEXPIRE key milliseconds
pub fn pexpire(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'pexpire' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let ms = match args[1].as_int() {
        Some(m) if m >= 0 => m as u64,
        _ => return RespValue::error("ERR invalid expire time"),
    };

    if db.expire(key, Duration::from_millis(ms)) {
        RespValue::integer(1)
    } else {
        RespValue::integer(0)
    }
}

/// TTL key
pub fn ttl(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'ttl' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.ttl(key) {
        Some(ttl) => RespValue::integer(ttl),
        None => RespValue::integer(-2),
    }
}

/// PTTL key
pub fn pttl(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'pttl' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.pttl(key) {
        Some(pttl) => RespValue::integer(pttl),
        None => RespValue::integer(-2),
    }
}

/// PERSIST key
pub fn persist(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'persist' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    if db.persist(key) {
        RespValue::integer(1)
    } else {
        RespValue::integer(0)
    }
}

/// TYPE key
pub fn key_type(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'type' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.key_type(key) {
        Some(t) => RespValue::SimpleString(t.to_string()),
        None => RespValue::SimpleString("none".to_string()),
    }
}

/// KEYS pattern
pub fn keys(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'keys' command");
    }

    let pattern = match args[0].as_str() {
        Some(p) => p,
        None => return RespValue::error("ERR invalid pattern"),
    };

    let keys = db.keys(pattern);
    let resp_keys: Vec<RespValue> = keys
        .into_iter()
        .map(|k| RespValue::bulk_string(k))
        .collect();

    RespValue::array(resp_keys)
}

/// DBSIZE
pub fn dbsize(_args: &[RespValue], db: &mut Database) -> RespValue {
    RespValue::integer(db.len() as i64)
}

/// FLUSHDB
pub fn flushdb(_args: &[RespValue], db: &mut Database) -> RespValue {
    db.flush();
    RespValue::ok()
}

/// RENAME key newkey
pub fn rename(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'rename' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let newkey = match args[1].as_str() {
        Some(k) => k.to_string(),
        None => return RespValue::error("ERR invalid new key"),
    };

    // Get the value
    let value = match db.get_string(key) {
        Some(v) => v,
        None => return RespValue::error("ERR no such key"),
    };

    // Delete old key and set new key
    db.delete(key);
    db.set_string(newkey, value);

    RespValue::ok()
}

/// RENAMENX key newkey
pub fn renamenx(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'renamenx' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let newkey = match args[1].as_str() {
        Some(k) => k.to_string(),
        None => return RespValue::error("ERR invalid new key"),
    };

    // Check if new key exists
    if db.exists(&newkey) {
        return RespValue::integer(0);
    }

    // Get the value
    let value = match db.get_string(key) {
        Some(v) => v,
        None => return RespValue::error("ERR no such key"),
    };

    // Delete old key and set new key
    db.delete(key);
    db.set_string(newkey, value);

    RespValue::integer(1)
}
