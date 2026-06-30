use std::time::Duration;

use crate::resp::RespValue;
use crate::storage::Database;

/// GET key
pub fn get(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'get' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.get_string(key) {
        Some(value) => RespValue::bulk(value),
        None => RespValue::null(),
    }
}

/// SET key value [EX seconds] [PX milliseconds] [NX|XX]
pub fn set(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'set' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k.to_string(),
        None => return RespValue::error("ERR invalid key"),
    };

    let value = match args[1].as_bytes() {
        Some(v) => v.to_vec(),
        None => return RespValue::error("ERR invalid value"),
    };

    let mut expire: Option<Duration> = None;
    let mut nx = false;
    let mut xx = false;

    // Parse options
    let mut i = 2;
    while i < args.len() {
        let opt = match args[i].as_str() {
            Some(s) => s.to_uppercase(),
            None => {
                i += 1;
                continue;
            }
        };

        match opt.as_str() {
            "EX" => {
                if i + 1 >= args.len() {
                    return RespValue::error("ERR syntax error");
                }
                let secs = match args[i + 1].as_int() {
                    Some(s) if s > 0 => s as u64,
                    _ => return RespValue::error("ERR invalid expire time"),
                };
                expire = Some(Duration::from_secs(secs));
                i += 2;
            }
            "PX" => {
                if i + 1 >= args.len() {
                    return RespValue::error("ERR syntax error");
                }
                let ms = match args[i + 1].as_int() {
                    Some(m) if m > 0 => m as u64,
                    _ => return RespValue::error("ERR invalid expire time"),
                };
                expire = Some(Duration::from_millis(ms));
                i += 2;
            }
            "NX" => {
                nx = true;
                i += 1;
            }
            "XX" => {
                xx = true;
                i += 1;
            }
            _ => {
                return RespValue::error("ERR syntax error");
            }
        }
    }

    // Check NX/XX conditions
    let exists = db.exists(&key);
    if nx && exists {
        return RespValue::null();
    }
    if xx && !exists {
        return RespValue::null();
    }

    // Set the value
    db.set_string(key.clone(), value);

    // Set expiration if specified
    if let Some(exp) = expire {
        db.expire(&key, exp);
    }

    RespValue::ok()
}

/// SETNX key value
pub fn setnx(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'setnx' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k.to_string(),
        None => return RespValue::error("ERR invalid key"),
    };

    let value = match args[1].as_bytes() {
        Some(v) => v.to_vec(),
        None => return RespValue::error("ERR invalid value"),
    };

    if db.setnx(key, value) {
        RespValue::integer(1)
    } else {
        RespValue::integer(0)
    }
}

/// SETEX key seconds value
pub fn setex(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 3 {
        return RespValue::error("ERR wrong number of arguments for 'setex' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k.to_string(),
        None => return RespValue::error("ERR invalid key"),
    };

    let seconds = match args[1].as_int() {
        Some(s) if s > 0 => s as u64,
        _ => return RespValue::error("ERR invalid expire time"),
    };

    let value = match args[2].as_bytes() {
        Some(v) => v.to_vec(),
        None => return RespValue::error("ERR invalid value"),
    };

    db.setex(key, seconds, value);
    RespValue::ok()
}

/// PSETEX key milliseconds value
pub fn psetex(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 3 {
        return RespValue::error("ERR wrong number of arguments for 'psetex' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k.to_string(),
        None => return RespValue::error("ERR invalid key"),
    };

    let ms = match args[1].as_int() {
        Some(m) if m > 0 => m as u64,
        _ => return RespValue::error("ERR invalid expire time"),
    };

    let value = match args[2].as_bytes() {
        Some(v) => v.to_vec(),
        None => return RespValue::error("ERR invalid value"),
    };

    db.set_string(key.clone(), value);
    db.expire(&key, Duration::from_millis(ms));
    RespValue::ok()
}

/// MGET key [key ...]
pub fn mget(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'mget' command");
    }

    let keys: Vec<String> = args
        .iter()
        .filter_map(|arg| arg.as_str().map(|s| s.to_string()))
        .collect();

    let values = db.mget(&keys);
    let resp_values: Vec<RespValue> = values
        .into_iter()
        .map(|v| match v {
            Some(data) => RespValue::bulk(data),
            None => RespValue::null(),
        })
        .collect();

    RespValue::array(resp_values)
}

/// MSET key value [key value ...]
pub fn mset(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.is_empty() || args.len() % 2 != 0 {
        return RespValue::error("ERR wrong number of arguments for 'mset' command");
    }

    let mut pairs = Vec::new();
    for chunk in args.chunks(2) {
        let key = match chunk[0].as_str() {
            Some(k) => k.to_string(),
            None => return RespValue::error("ERR invalid key"),
        };
        let value = match chunk[1].as_bytes() {
            Some(v) => v.to_vec(),
            None => return RespValue::error("ERR invalid value"),
        };
        pairs.push((key, value));
    }

    db.mset(pairs);
    RespValue::ok()
}

/// APPEND key value
pub fn append(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'append' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let value = match args[1].as_bytes() {
        Some(v) => v,
        None => return RespValue::error("ERR invalid value"),
    };

    match db.append(key, value) {
        Ok(len) => RespValue::integer(len as i64),
        Err(e) => RespValue::error(e),
    }
}

/// STRLEN key
pub fn strlen(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'strlen' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.strlen(key) {
        Ok(len) => RespValue::integer(len as i64),
        Err(e) => RespValue::error(e),
    }
}

/// INCR key
pub fn incr(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'incr' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.incr(key, 1) {
        Ok(value) => RespValue::integer(value),
        Err(e) => RespValue::error(e),
    }
}

/// INCRBY key increment
pub fn incrby(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'incrby' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let increment = match args[1].as_int() {
        Some(i) => i,
        None => return RespValue::error("ERR value is not an integer"),
    };

    match db.incr(key, increment) {
        Ok(value) => RespValue::integer(value),
        Err(e) => RespValue::error(e),
    }
}

/// DECR key
pub fn decr(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'decr' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.incr(key, -1) {
        Ok(value) => RespValue::integer(value),
        Err(e) => RespValue::error(e),
    }
}

/// DECRBY key decrement
pub fn decrby(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'decrby' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let decrement = match args[1].as_int() {
        Some(i) => i,
        None => return RespValue::error("ERR value is not an integer"),
    };

    match db.incr(key, -decrement) {
        Ok(value) => RespValue::integer(value),
        Err(e) => RespValue::error(e),
    }
}

/// GETSET key value
pub fn getset(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'getset' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let value = match args[1].as_bytes() {
        Some(v) => v.to_vec(),
        None => return RespValue::error("ERR invalid value"),
    };

    let old_value = db.get_string(key);
    db.set_string(key.to_string(), value);

    match old_value {
        Some(data) => RespValue::bulk(data),
        None => RespValue::null(),
    }
}
