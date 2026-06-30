use crate::resp::RespValue;
use crate::storage::{Database, RedisObject, ZSetObject};

/// Get sorted set from database, creating if necessary
fn get_or_create_zset<'a>(db: &'a mut Database, key: &str) -> Result<&'a mut ZSetObject, RespValue> {
    let key_string = key.to_string();

    if !db.exists(key) {
        db.set(key_string.clone(), RedisObject::ZSet(ZSetObject::new()));
    }

    match db.get_mut(key) {
        Some(RedisObject::ZSet(zset)) => Ok(zset),
        Some(_) => Err(RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value")),
        None => Err(RespValue::error("ERR key does not exist")),
    }
}

/// ZADD key [NX|XX] [GT|LT] [CH] score member [score member ...]
pub fn zadd(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() < 3 {
        return RespValue::error("ERR wrong number of arguments for 'zadd' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    // Parse options and find start of score-member pairs
    let mut nx = false;
    let mut xx = false;
    let mut ch = false;
    let mut idx = 1;

    while idx < args.len() {
        let opt = match args[idx].as_str() {
            Some(s) => s.to_uppercase(),
            None => break,
        };

        match opt.as_str() {
            "NX" => {
                nx = true;
                idx += 1;
            }
            "XX" => {
                xx = true;
                idx += 1;
            }
            "CH" => {
                ch = true;
                idx += 1;
            }
            _ => break,
        }
    }

    // Check we have score-member pairs
    if (args.len() - idx) % 2 != 0 || args.len() - idx < 2 {
        return RespValue::error("ERR syntax error");
    }

    let zset = match get_or_create_zset(db, key) {
        Ok(z) => z,
        Err(e) => return e,
    };

    let mut added = 0;
    let mut changed = 0;

    for chunk in args[idx..].chunks(2) {
        let score = match chunk[0].as_str().and_then(|s| s.parse::<f64>().ok()) {
            Some(s) => s,
            None => return RespValue::error("ERR value is not a valid float"),
        };

        let member = match chunk[1].as_bytes() {
            Some(m) => m.to_vec(),
            None => return RespValue::error("ERR invalid member"),
        };

        let exists = zset.dict.contains_key(&member);
        let old_score = zset.score(&member);

        // Apply NX/XX
        if nx && exists {
            continue;
        }
        if xx && !exists {
            continue;
        }

        let is_new = zset.add(score, member);
        if is_new {
            added += 1;
        } else if old_score != Some(score) {
            changed += 1;
        }
    }

    if ch {
        RespValue::integer(added + changed)
    } else {
        RespValue::integer(added)
    }
}

/// ZREM key member [member ...]
pub fn zrem(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'zrem' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.get_mut(key) {
        Some(RedisObject::ZSet(zset)) => {
            let mut removed = 0;
            for arg in &args[1..] {
                if let Some(member) = arg.as_bytes() {
                    if zset.remove(member) {
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

/// ZSCORE key member
pub fn zscore(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'zscore' command");
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
        Some(RedisObject::ZSet(zset)) => {
            match zset.score(member) {
                Some(score) => RespValue::bulk_string(score.to_string()),
                None => RespValue::null(),
            }
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::null(),
    }
}

/// ZCARD key
pub fn zcard(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'zcard' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    match db.get(key) {
        Some(RedisObject::ZSet(zset)) => RespValue::integer(zset.len() as i64),
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::integer(0),
    }
}

/// ZRANGE key start stop [WITHSCORES]
pub fn zrange(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() < 3 {
        return RespValue::error("ERR wrong number of arguments for 'zrange' command");
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

    let withscores = args.len() > 3 && args[3].as_str()
        .map(|s| s.eq_ignore_ascii_case("WITHSCORES"))
        .unwrap_or(false);

    match db.get(key) {
        Some(RedisObject::ZSet(zset)) => {
            let len = zset.len() as i64;
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

            let mut result = Vec::new();
            for (score, member) in &zset.sorted[start..stop] {
                result.push(RespValue::bulk(member.clone()));
                if withscores {
                    result.push(RespValue::bulk_string(score.to_string()));
                }
            }

            RespValue::array(result)
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::array(vec![]),
    }
}

/// ZRANK key member
pub fn zrank(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 2 {
        return RespValue::error("ERR wrong number of arguments for 'zrank' command");
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
        Some(RedisObject::ZSet(zset)) => {
            // Find position in sorted list
            for (i, (_, m)) in zset.sorted.iter().enumerate() {
                if m == member {
                    return RespValue::integer(i as i64);
                }
            }
            RespValue::null()
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::null(),
    }
}

/// ZCOUNT key min max
pub fn zcount(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 3 {
        return RespValue::error("ERR wrong number of arguments for 'zcount' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let min = match parse_score_bound(args[1].as_str().unwrap_or("")) {
        Some(s) => s,
        None => return RespValue::error("ERR min or max is not a float"),
    };

    let max = match parse_score_bound(args[2].as_str().unwrap_or("")) {
        Some(s) => s,
        None => return RespValue::error("ERR min or max is not a float"),
    };

    match db.get(key) {
        Some(RedisObject::ZSet(zset)) => {
            let count = zset.sorted.iter()
                .filter(|(score, _)| *score >= min && *score <= max)
                .count();
            RespValue::integer(count as i64)
        }
        Some(_) => RespValue::error("WRONGTYPE Operation against a key holding the wrong kind of value"),
        None => RespValue::integer(0),
    }
}

/// ZINCRBY key increment member
pub fn zincrby(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() != 3 {
        return RespValue::error("ERR wrong number of arguments for 'zincrby' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let increment = match args[1].as_str().and_then(|s| s.parse::<f64>().ok()) {
        Some(i) => i,
        None => return RespValue::error("ERR value is not a valid float"),
    };

    let member = match args[2].as_bytes() {
        Some(m) => m.to_vec(),
        None => return RespValue::error("ERR invalid member"),
    };

    let zset = match get_or_create_zset(db, key) {
        Ok(z) => z,
        Err(e) => return e,
    };

    let new_score = zset.score(&member).unwrap_or(0.0) + increment;
    zset.add(new_score, member);

    RespValue::bulk_string(new_score.to_string())
}

/// Parse score bound (handle -inf, +inf, exclusive)
fn parse_score_bound(s: &str) -> Option<f64> {
    match s.to_lowercase().as_str() {
        "-inf" => Some(f64::NEG_INFINITY),
        "+inf" | "inf" => Some(f64::INFINITY),
        _ => {
            if s.starts_with('(') {
                // Exclusive bound - for simplicity, treat as inclusive
                s[1..].parse().ok()
            } else {
                s.parse().ok()
            }
        }
    }
}
