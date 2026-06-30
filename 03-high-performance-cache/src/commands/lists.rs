use crate::resp::RespValue;
use crate::storage::{Database, RedisObject};

/// Get list from database, creating if necessary
fn get_or_create_list<'a>(db: &'a mut Database, key: &str) -> Result<&'a mut Vec<Vec<u8>>, RespValue> {
    let key_string = key.to_string();

    if !db.exists(key) {
        db.set(key_string.clone(), RedisObject::List(Vec::new()));
    }

    match db.get_mut(key) {
        Some(RedisObject::List(list)) => Ok(list),
        Some(_) => Err(RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value")),
        None => Err(RespValue::error("ERR key does not exist")),
    }
}

/// LPUSH key element [element ...]
pub fn lpush(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'lpush' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let list = match get_or_create_list(db, key) {
        Ok(l) => l,
        Err(e) => return e,
    };

    for arg in &args[1..] {
        if let Some(value) = arg.as_bytes() {
            list.insert(0, value.to_vec());
        }
    }

    RespValue::integer(list.len() as i64)
}

/// RPUSH key element [element ...]
pub fn rpush(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'rpush' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let list = match get_or_create_list(db, key) {
        Ok(l) => l,
        Err(e) => return e,
    };

    for arg in &args[1..] {
        if let Some(value) = arg.as_bytes() {
            list.push(value.to_vec());
        }
    }

    RespValue::integer(list.len() as i64)
}

/// LPOP key [count]
pub fn lpop(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'lpop' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let count = if args.len() > 1 {
        match args[1].as_int() {
            Some(c) if c > 0 => Some(c as usize),
            Some(_) => return RespValue::error("ERR value is out of range"),
            None => return RespValue::error("ERR value is not an integer"),
        }
    } else {
        None
    };

    match db.get_mut(key) {
        Some(RedisObject::List(list)) => {
            if list.is_empty() {
                return RespValue::null();
            }

            if let Some(count) = count {
                let mut result = Vec::new();
                for _ in 0..count {
                    if list.is_empty() {
                        break;
                    }
                    result.push(RespValue::bulk(list.remove(0)));
                }
                RespValue::array(result)
            } else {
                RespValue::bulk(list.remove(0))
            }
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::null(),
    }
}

/// RPOP key [count]
pub fn rpop(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'rpop' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let count = if args.len() > 1 {
        match args[1].as_int() {
            Some(c) if c > 0 => Some(c as usize),
            Some(_) => return RespValue::error("ERR value is out of range"),
            None => return RespValue::error("ERR value is not an integer"),
        }
    } else {
        None
    };

    match db.get_mut(key) {
        Some(RedisObject::List(list)) => {
            if list.is_empty() {
                return RespValue::null();
            }

            if let Some(count) = count {
                let mut result = Vec::new();
                for _ in 0..count {
                    if list.is_empty() {
                        break;
                    }
                    result.push(RespValue::bulk(list.pop().unwrap()));
                }
                RespValue::array(result)
            } else {
                RespValue::bulk(list.pop().unwrap())
            }
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::null(),
    }
}

/// LLEN key
pub fn llen(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'llen' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.get(key) {
        Some(RedisObject::List(list)) => RespValue::integer(list.len() as i64),
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::integer(0),
    }
}

/// LRANGE key start stop
pub fn lrange(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 3 {
        return RespValue::error("ERR wrong number of arguments for 'lrange' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let start = match args[1].as_int() {
        Some(s) => s,
        None => return RespValue::error("ERR value is not an integer"),
    };

    let stop = match args[2].as_int() {
        Some(s) => s,
        None => return RespValue::error("ERR value is not an integer"),
    };

    match db.get(key) {
        Some(RedisObject::List(list)) => {
            let len = list.len() as i64;
            if len == 0 {
                return RespValue::array(vec![]);
            }

            // Convert negative indices
            let start = if start < 0 {
                (len + start).max(0) as usize
            } else {
                start.min(len) as usize
            };

            let stop = if stop < 0 {
                (len + stop + 1).max(0) as usize
            } else {
                (stop + 1).min(len) as usize
            };

            if start >= stop {
                return RespValue::array(vec![]);
            }

            let result: Vec<RespValue> = list[start..stop]
                .iter()
                .map(|v| RespValue::bulk(v.clone()))
                .collect();

            RespValue::array(result)
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::array(vec![]),
    }
}

/// LINDEX key index
pub fn lindex(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'lindex' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let index = match args[1].as_int() {
        Some(i) => i,
        None => return RespValue::error("ERR value is not an integer"),
    };

    match db.get(key) {
        Some(RedisObject::List(list)) => {
            let len = list.len() as i64;
            let idx = if index < 0 {
                len + index
            } else {
                index
            };

            if idx < 0 || idx >= len {
                return RespValue::null();
            }

            RespValue::bulk(list[idx as usize].clone())
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::null(),
    }
}

/// LSET key index element
pub fn lset(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 3 {
        return RespValue::error("ERR wrong number of arguments for 'lset' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let index = match args[1].as_int() {
        Some(i) => i,
        None => return RespValue::error("ERR value is not an integer"),
    };

    let value = match args[2].as_bytes() {
        Some(v) => v.to_vec(),
        None => return RespValue::error("ERR invalid value"),
    };

    match db.get_mut(key) {
        Some(RedisObject::List(list)) => {
            let len = list.len() as i64;
            let idx = if index < 0 {
                len + index
            } else {
                index
            };

            if idx < 0 || idx >= len {
                return RespValue::error("ERR index out of range");
            }

            list[idx as usize] = value;
            RespValue::ok()
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::error("ERR no such key"),
    }
}
