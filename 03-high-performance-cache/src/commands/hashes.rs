use std::collections::HashMap;

use crate::resp::RespValue;
use crate::storage::{Database, RedisObject};

/// Get hash from database, creating if necessary
fn get_or_create_hash<'a>(db: &'a mut Database, key: &str) -> Result<&'a mut HashMap<Vec<u8>, Vec<u8>>, RespValue> {
    let key_string = key.to_string();

    if !db.exists(key) {
        db.set(key_string.clone(), RedisObject::Hash(HashMap::new()));
    }

    match db.get_mut(key) {
        Some(RedisObject::Hash(hash)) => Ok(hash),
        Some(_) => Err(RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value")),
        None => Err(RespValue::error("ERR key does not exist")),
    }
}

/// HSET key field value [field value ...]
pub fn hset(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() < 3 || args.len() % 2 == 0 {
        return RespValue::error("ERR wrong number of arguments for 'hset' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let hash = match get_or_create_hash(db, key) {
        Ok(h) => h,
        Err(e) => return e,
    };

    let mut added = 0;
    for chunk in args[1..].chunks(2) {
        let field = match chunk[0].as_bytes() {
            Some(f) => f.to_vec(),
            None => return RespValue::error("ERR invalid field"),
        };
        let value = match chunk[1].as_bytes() {
            Some(v) => v.to_vec(),
            None => return RespValue::error("ERR invalid value"),
        };

        if hash.insert(field, value).is_none() {
            added += 1;
        }
    }

    RespValue::integer(added)
}

/// HGET key field
pub fn hget(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'hget' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let field = match args[1].as_bytes() {
        Some(f) => f,
        None => return RespValue::error("ERR invalid field"),
    };

    match db.get(key) {
        Some(RedisObject::Hash(hash)) => {
            match hash.get(field) {
                Some(value) => RespValue::bulk(value.clone()),
                None => RespValue::null(),
            }
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::null(),
    }
}

/// HDEL key field [field ...]
pub fn hdel(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'hdel' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.get_mut(key) {
        Some(RedisObject::Hash(hash)) => {
            let mut deleted = 0;
            for arg in &args[1..] {
                if let Some(field) = arg.as_bytes() {
                    if hash.remove(field).is_some() {
                        deleted += 1;
                    }
                }
            }
            RespValue::integer(deleted)
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::integer(0),
    }
}

/// HEXISTS key field
pub fn hexists(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'hexists' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let field = match args[1].as_bytes() {
        Some(f) => f,
        None => return RespValue::error("ERR invalid field"),
    };

    match db.get(key) {
        Some(RedisObject::Hash(hash)) => {
            if hash.contains_key(field) {
                RespValue::integer(1)
            } else {
                RespValue::integer(0)
            }
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::integer(0),
    }
}

/// HLEN key
pub fn hlen(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'hlen' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.get(key) {
        Some(RedisObject::Hash(hash)) => RespValue::integer(hash.len() as i64),
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::integer(0),
    }
}

/// HGETALL key
pub fn hgetall(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'hgetall' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.get(key) {
        Some(RedisObject::Hash(hash)) => {
            let mut result = Vec::with_capacity(hash.len() * 2);
            for (field, value) in hash {
                result.push(RespValue::bulk(field.clone()));
                result.push(RespValue::bulk(value.clone()));
            }
            RespValue::array(result)
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::array(vec![]),
    }
}

/// HKEYS key
pub fn hkeys(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'hkeys' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.get(key) {
        Some(RedisObject::Hash(hash)) => {
            let keys: Vec<RespValue> = hash
                .keys()
                .map(|k| RespValue::bulk(k.clone()))
                .collect();
            RespValue::array(keys)
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::array(vec![]),
    }
}

/// HVALS key
pub fn hvals(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'hvals' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.get(key) {
        Some(RedisObject::Hash(hash)) => {
            let values: Vec<RespValue> = hash
                .values()
                .map(|v| RespValue::bulk(v.clone()))
                .collect();
            RespValue::array(values)
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::array(vec![]),
    }
}

/// HMSET key field value [field value ...] (deprecated, use HSET)
pub fn hmset(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() < 3 || args.len() % 2 == 0 {
        return RespValue::error("ERR wrong number of arguments for 'hmset' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let hash = match get_or_create_hash(db, key) {
        Ok(h) => h,
        Err(e) => return e,
    };

    for chunk in args[1..].chunks(2) {
        let field = match chunk[0].as_bytes() {
            Some(f) => f.to_vec(),
            None => return RespValue::error("ERR invalid field"),
        };
        let value = match chunk[1].as_bytes() {
            Some(v) => v.to_vec(),
            None => return RespValue::error("ERR invalid value"),
        };
        hash.insert(field, value);
    }

    RespValue::ok()
}

/// HMGET key field [field ...]
pub fn hmget(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'hmget' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.get(key) {
        Some(RedisObject::Hash(hash)) => {
            let values: Vec<RespValue> = args[1..]
                .iter()
                .map(|arg| {
                    if let Some(field) = arg.as_bytes() {
                        match hash.get(field) {
                            Some(value) => RespValue::bulk(value.clone()),
                            None => RespValue::null(),
                        }
                    } else {
                        RespValue::null()
                    }
                })
                .collect();
            RespValue::array(values)
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => {
            let nulls: Vec<RespValue> = args[1..].iter().map(|_| RespValue::null()).collect();
            RespValue::array(nulls)
        }
    }
}
