use std::collections::HashSet;

use crate::resp::RespValue;
use crate::storage::{Database, RedisObject};

/// Get set from database, creating if necessary
fn get_or_create_set<'a>(db: &'a mut Database, key: &str) -> Result<&'a mut HashSet<Vec<u8>>, RespValue> {
    let key_string = key.to_string();

    if !db.exists(key) {
        db.set(key_string.clone(), RedisObject::Set(HashSet::new()));
    }

    match db.get_mut(key) {
        Some(RedisObject::Set(set)) => Ok(set),
        Some(_) => Err(RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value")),
        None => Err(RespValue::error("ERR key does not exist")),
    }
}

/// SADD key member [member ...]
pub fn sadd(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'sadd' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let set = match get_or_create_set(db, key) {
        Ok(s) => s,
        Err(e) => return e,
    };

    let mut added = 0;
    for arg in &args[1..] {
        if let Some(member) = arg.as_bytes() {
            if set.insert(member.to_vec()) {
                added += 1;
            }
        }
    }

    RespValue::integer(added)
}

/// SREM key member [member ...]
pub fn srem(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'srem' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.get_mut(key) {
        Some(RedisObject::Set(set)) => {
            let mut removed = 0;
            for arg in &args[1..] {
                if let Some(member) = arg.as_bytes() {
                    if set.remove(member) {
                        removed += 1;
                    }
                }
            }
            RespValue::integer(removed)
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::integer(0),
    }
}

/// SMEMBERS key
pub fn smembers(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'smembers' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.get(key) {
        Some(RedisObject::Set(set)) => {
            let members: Vec<RespValue> = set
                .iter()
                .map(|m| RespValue::bulk(m.clone()))
                .collect();
            RespValue::array(members)
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::array(vec![]),
    }
}

/// SISMEMBER key member
pub fn sismember(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'sismember' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let member = match args[1].as_bytes() {
        Some(m) => m,
        None => return RespValue::error("ERR invalid member"),
    };

    match db.get(key) {
        Some(RedisObject::Set(set)) => {
            if set.contains(member) {
                RespValue::integer(1)
            } else {
                RespValue::integer(0)
            }
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::integer(0),
    }
}

/// SCARD key
pub fn scard(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'scard' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.get(key) {
        Some(RedisObject::Set(set)) => RespValue::integer(set.len() as i64),
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::integer(0),
    }
}

/// SINTER key [key ...]
pub fn sinter(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'sinter' command");
    }

    let mut result: Option<HashSet<Vec<u8>>> = None;

    for arg in args {
        let key = match arg.as_str() {
            Some(k) => k,
            None => return RespValue::error("ERR invalid key"),
        };

        let set = match db.get(key) {
            Some(RedisObject::Set(s)) => s.clone(),
            Some(_) => return RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
            None => return RespValue::array(vec![]),
        };

        result = match result {
            Some(r) => Some(r.intersection(&set).cloned().collect()),
            None => Some(set),
        };
    }

    let members: Vec<RespValue> = result
        .unwrap_or_default()
        .into_iter()
        .map(|m| RespValue::bulk(m))
        .collect();

    RespValue::array(members)
}

/// SUNION key [key ...]
pub fn sunion(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'sunion' command");
    }

    let mut result: HashSet<Vec<u8>> = HashSet::new();

    for arg in args {
        let key = match arg.as_str() {
            Some(k) => k,
            None => return RespValue::error("ERR invalid key"),
        };

        match db.get(key) {
            Some(RedisObject::Set(set)) => {
                result.extend(set.iter().cloned());
            }
            Some(_) => return RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
            None => {}
        }
    }

    let members: Vec<RespValue> = result
        .into_iter()
        .map(|m| RespValue::bulk(m))
        .collect();

    RespValue::array(members)
}

/// SDIFF key [key ...]
pub fn sdiff(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'sdiff' command");
    }

    let first_key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let mut result = match db.get(first_key) {
        Some(RedisObject::Set(s)) => s.clone(),
        Some(_) => return RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => return RespValue::array(vec![]),
    };

    for arg in &args[1..] {
        let key = match arg.as_str() {
            Some(k) => k,
            None => return RespValue::error("ERR invalid key"),
        };

        if let Some(RedisObject::Set(set)) = db.get(key) {
            result = result.difference(set).cloned().collect();
        }
    }

    let members: Vec<RespValue> = result
        .into_iter()
        .map(|m| RespValue::bulk(m))
        .collect();

    RespValue::array(members)
}
