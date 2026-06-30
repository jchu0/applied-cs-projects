//! Redis Streams commands (XADD, XREAD, XRANGE, XLEN, etc.)

use crate::resp::RespValue;
use crate::storage::streams::{Stream, StreamId, StreamError};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Stream storage (separate from main database due to different structure)
pub type StreamStore = Arc<RwLock<HashMap<String, Stream>>>;

/// Create a new stream store
pub fn new_stream_store() -> StreamStore {
    Arc::new(RwLock::new(HashMap::new()))
}

/// Helper to get or create a stream
fn get_or_create_stream(store: &StreamStore, key: &str) -> Stream {
    let mut streams = store.write().unwrap();
    streams.entry(key.to_string()).or_insert_with(Stream::new).clone()
}

/// XADD key [NOMKSTREAM] [MAXLEN|MINID [=|~] threshold] *|id field value [field value ...]
pub fn xadd(args: &[RespValue], store: &StreamStore) -> RespValue {
    if args.len() < 4 {
        return RespValue::error("ERR wrong number of arguments for 'xadd' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k.to_string(),
        None => return RespValue::error("ERR invalid key"),
    };

    let mut idx = 1;
    let mut no_mkstream = false;
    let mut maxlen: Option<usize> = None;

    // Parse options
    while idx < args.len() {
        match args[idx].as_str().map(|s| s.to_uppercase()).as_deref() {
            Some("NOMKSTREAM") => {
                no_mkstream = true;
                idx += 1;
            }
            Some("MAXLEN") => {
                idx += 1;
                // Skip ~ or = if present
                if let Some(s) = args.get(idx).and_then(|v| v.as_str()) {
                    if s == "~" || s == "=" {
                        idx += 1;
                    }
                }
                if let Some(len) = args.get(idx).and_then(|v| v.as_int()) {
                    maxlen = Some(len as usize);
                    idx += 1;
                } else {
                    return RespValue::error("ERR value is not an integer or out of range");
                }
            }
            Some("MINID") => {
                // Skip MINID option for now
                idx += 1;
                if let Some(s) = args.get(idx).and_then(|v| v.as_str()) {
                    if s == "~" || s == "=" {
                        idx += 1;
                    }
                }
                idx += 1; // Skip the threshold
            }
            _ => break,
        }
    }

    // Get ID
    let id_str = match args.get(idx).and_then(|v| v.as_str()) {
        Some(id) => id,
        None => return RespValue::error("ERR wrong number of arguments for 'xadd' command"),
    };
    idx += 1;

    // Parse field-value pairs
    let remaining = &args[idx..];
    if remaining.len() % 2 != 0 {
        return RespValue::error("ERR wrong number of arguments for 'xadd' command");
    }

    let mut fields = Vec::new();
    for pair in remaining.chunks(2) {
        let field = match pair[0].as_bytes() {
            Some(f) => f.to_vec(),
            None => return RespValue::error("ERR invalid field"),
        };
        let value = match pair[1].as_bytes() {
            Some(v) => v.to_vec(),
            None => return RespValue::error("ERR invalid value"),
        };
        fields.push((field, value));
    }

    if fields.is_empty() {
        return RespValue::error("ERR wrong number of arguments for 'xadd' command");
    }

    let mut streams = store.write().unwrap();

    // Check NOMKSTREAM
    if no_mkstream && !streams.contains_key(&key) {
        return RespValue::null();
    }

    let stream = streams.entry(key).or_insert_with(Stream::new);

    if let Some(len) = maxlen {
        stream.set_max_len(Some(len));
    }

    match stream.add(id_str, fields) {
        Ok(id) => RespValue::bulk_string(id.to_string()),
        Err(e) => RespValue::error(e.to_string()),
    }
}

/// XLEN key
pub fn xlen(args: &[RespValue], store: &StreamStore) -> RespValue {
    if args.len() != 1 {
        return RespValue::error("ERR wrong number of arguments for 'xlen' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let streams = store.read().unwrap();
    let len = streams.get(key).map_or(0, |s| s.len());
    RespValue::integer(len as i64)
}

/// XRANGE key start end [COUNT count]
pub fn xrange(args: &[RespValue], store: &StreamStore) -> RespValue {
    if args.len() < 3 {
        return RespValue::error("ERR wrong number of arguments for 'xrange' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let start = match args[1].as_str() {
        Some("-") => StreamId::min(),
        Some(s) => match StreamId::parse(s, None) {
            Ok(id) => id,
            Err(e) => return RespValue::error(e.to_string()),
        },
        None => return RespValue::error("ERR invalid start ID"),
    };

    let end = match args[2].as_str() {
        Some("+") => StreamId::max(),
        Some(s) => match StreamId::parse(s, None) {
            Ok(id) => id,
            Err(e) => return RespValue::error(e.to_string()),
        },
        None => return RespValue::error("ERR invalid end ID"),
    };

    let count = if args.len() > 4 {
        match args[3].as_str().map(|s| s.to_uppercase()).as_deref() {
            Some("COUNT") => args.get(4).and_then(|v| v.as_int()).map(|c| c as usize),
            _ => None,
        }
    } else {
        None
    };

    let streams = store.read().unwrap();
    let entries = match streams.get(key) {
        Some(stream) => stream.range(start, end, count),
        None => return RespValue::array(vec![]),
    };

    let result: Vec<RespValue> = entries
        .iter()
        .map(|entry| {
            let fields: Vec<RespValue> = entry
                .fields
                .iter()
                .flat_map(|(k, v)| vec![RespValue::bulk(k.clone()), RespValue::bulk(v.clone())])
                .collect();
            RespValue::array(vec![
                RespValue::bulk_string(entry.id.to_string()),
                RespValue::array(fields),
            ])
        })
        .collect();

    RespValue::array(result)
}

/// XREVRANGE key end start [COUNT count]
pub fn xrevrange(args: &[RespValue], store: &StreamStore) -> RespValue {
    if args.len() < 3 {
        return RespValue::error("ERR wrong number of arguments for 'xrevrange' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let end = match args[1].as_str() {
        Some("+") => StreamId::max(),
        Some(s) => match StreamId::parse(s, None) {
            Ok(id) => id,
            Err(e) => return RespValue::error(e.to_string()),
        },
        None => return RespValue::error("ERR invalid end ID"),
    };

    let start = match args[2].as_str() {
        Some("-") => StreamId::min(),
        Some(s) => match StreamId::parse(s, None) {
            Ok(id) => id,
            Err(e) => return RespValue::error(e.to_string()),
        },
        None => return RespValue::error("ERR invalid start ID"),
    };

    let count = if args.len() > 4 {
        match args[3].as_str().map(|s| s.to_uppercase()).as_deref() {
            Some("COUNT") => args.get(4).and_then(|v| v.as_int()).map(|c| c as usize),
            _ => None,
        }
    } else {
        None
    };

    let streams = store.read().unwrap();
    let entries = match streams.get(key) {
        Some(stream) => stream.revrange(end, start, count),
        None => return RespValue::array(vec![]),
    };

    let result: Vec<RespValue> = entries
        .iter()
        .map(|entry| {
            let fields: Vec<RespValue> = entry
                .fields
                .iter()
                .flat_map(|(k, v)| vec![RespValue::bulk(k.clone()), RespValue::bulk(v.clone())])
                .collect();
            RespValue::array(vec![
                RespValue::bulk_string(entry.id.to_string()),
                RespValue::array(fields),
            ])
        })
        .collect();

    RespValue::array(result)
}

/// XREAD [COUNT count] [BLOCK milliseconds] STREAMS key [key ...] id [id ...]
pub fn xread(args: &[RespValue], store: &StreamStore) -> RespValue {
    if args.len() < 3 {
        return RespValue::error("ERR wrong number of arguments for 'xread' command");
    }

    let mut idx = 0;
    let mut count: Option<usize> = None;
    let mut _block: Option<u64> = None;

    // Parse options
    while idx < args.len() {
        match args[idx].as_str().map(|s| s.to_uppercase()).as_deref() {
            Some("COUNT") => {
                idx += 1;
                count = args.get(idx).and_then(|v| v.as_int()).map(|c| c as usize);
                idx += 1;
            }
            Some("BLOCK") => {
                idx += 1;
                _block = args.get(idx).and_then(|v| v.as_int()).map(|b| b as u64);
                idx += 1;
            }
            Some("STREAMS") => {
                idx += 1;
                break;
            }
            _ => {
                idx += 1;
            }
        }
    }

    // Parse keys and IDs
    let remaining = &args[idx..];
    if remaining.is_empty() || remaining.len() % 2 != 0 {
        return RespValue::error("ERR Unbalanced 'xread' list of streams");
    }

    let half = remaining.len() / 2;
    let keys: Vec<String> = remaining[..half]
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    let ids: Vec<&str> = remaining[half..]
        .iter()
        .filter_map(|v| v.as_str())
        .collect();

    if keys.len() != ids.len() {
        return RespValue::error("ERR Unbalanced 'xread' list of streams");
    }

    let streams = store.read().unwrap();
    let mut result = Vec::new();

    for (key, id_str) in keys.iter().zip(ids.iter()) {
        if let Some(stream) = streams.get(key) {
            let start_id = if *id_str == "$" {
                stream.last_id()
            } else {
                match StreamId::parse(id_str, None) {
                    Ok(id) => id,
                    Err(e) => return RespValue::error(e.to_string()),
                }
            };

            let entries = stream.read_after(start_id, count);
            if !entries.is_empty() {
                let entry_values: Vec<RespValue> = entries
                    .iter()
                    .map(|entry| {
                        let fields: Vec<RespValue> = entry
                            .fields
                            .iter()
                            .flat_map(|(k, v)| {
                                vec![RespValue::bulk(k.clone()), RespValue::bulk(v.clone())]
                            })
                            .collect();
                        RespValue::array(vec![
                            RespValue::bulk_string(entry.id.to_string()),
                            RespValue::array(fields),
                        ])
                    })
                    .collect();

                result.push(RespValue::array(vec![
                    RespValue::bulk_string(key.clone()),
                    RespValue::array(entry_values),
                ]));
            }
        }
    }

    if result.is_empty() {
        RespValue::null()
    } else {
        RespValue::array(result)
    }
}

/// XTRIM key MAXLEN|MINID [=|~] threshold
pub fn xtrim(args: &[RespValue], store: &StreamStore) -> RespValue {
    if args.len() < 3 {
        return RespValue::error("ERR wrong number of arguments for 'xtrim' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k.to_string(),
        None => return RespValue::error("ERR invalid key"),
    };

    let strategy = match args[1].as_str().map(|s| s.to_uppercase()).as_deref() {
        Some("MAXLEN") => "MAXLEN",
        Some("MINID") => return RespValue::error("ERR MINID trimming not supported"),
        _ => return RespValue::error("ERR syntax error"),
    };

    let mut idx = 2;
    // Skip ~ or = if present
    if let Some(s) = args.get(idx).and_then(|v| v.as_str()) {
        if s == "~" || s == "=" {
            idx += 1;
        }
    }

    let threshold = match args.get(idx).and_then(|v| v.as_int()) {
        Some(t) if t >= 0 => t as usize,
        _ => return RespValue::error("ERR value is not an integer or out of range"),
    };

    let mut streams = store.write().unwrap();
    let removed = match streams.get_mut(&key) {
        Some(stream) => {
            if strategy == "MAXLEN" {
                stream.trim(threshold)
            } else {
                0
            }
        }
        None => 0,
    };

    RespValue::integer(removed as i64)
}

/// XDEL key id [id ...]
pub fn xdel(args: &[RespValue], store: &StreamStore) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'xdel' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k.to_string(),
        None => return RespValue::error("ERR invalid key"),
    };

    let ids: Result<Vec<StreamId>, _> = args[1..]
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or(StreamError::InvalidId)
                .and_then(|s| StreamId::parse(s, None))
        })
        .collect();

    let ids = match ids {
        Ok(ids) => ids,
        Err(e) => return RespValue::error(e.to_string()),
    };

    let mut streams = store.write().unwrap();
    let deleted = match streams.get_mut(&key) {
        Some(stream) => stream.delete(&ids),
        None => 0,
    };

    RespValue::integer(deleted as i64)
}

/// XINFO STREAM key
pub fn xinfo_stream(args: &[RespValue], store: &StreamStore) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'xinfo' command");
    }

    let subcommand = match args[0].as_str().map(|s| s.to_uppercase()) {
        Some(s) => s,
        None => return RespValue::error("ERR invalid subcommand"),
    };

    if subcommand != "STREAM" {
        return RespValue::error("ERR unknown subcommand or wrong number of arguments");
    }

    let key = match args[1].as_str() {
        Some(k) => k,
        None => return RespValue::error("ERR invalid key"),
    };

    let streams = store.read().unwrap();
    let stream = match streams.get(key) {
        Some(s) => s,
        None => return RespValue::error("ERR no such key"),
    };

    let info = stream.info();

    RespValue::array(vec![
        RespValue::bulk_string("length"),
        RespValue::integer(info.length as i64),
        RespValue::bulk_string("radix-tree-keys"),
        RespValue::integer(info.radix_tree_keys as i64),
        RespValue::bulk_string("radix-tree-nodes"),
        RespValue::integer(info.radix_tree_nodes as i64),
        RespValue::bulk_string("last-generated-id"),
        RespValue::bulk_string(info.last_generated_id.to_string()),
        RespValue::bulk_string("groups"),
        RespValue::integer(info.groups_count as i64),
    ])
}

/// Alias for xinfo_stream
pub fn xinfo(args: &[RespValue], store: &StreamStore) -> RespValue {
    xinfo_stream(args, store)
}

/// XGROUP CREATE key groupname id [MKSTREAM]
pub fn xgroup(args: &[RespValue], store: &StreamStore) -> RespValue {
    if args.len() < 2 {
        return RespValue::error("ERR wrong number of arguments for 'xgroup' command");
    }

    let subcommand = match args[0].as_str().map(|s| s.to_uppercase()) {
        Some(s) => s,
        None => return RespValue::error("ERR invalid subcommand"),
    };

    match subcommand.as_str() {
        "CREATE" => {
            if args.len() < 4 {
                return RespValue::error("ERR wrong number of arguments for 'xgroup create'");
            }

            let key = match args[1].as_str() {
                Some(k) => k.to_string(),
                None => return RespValue::error("ERR invalid key"),
            };

            let groupname = match args[2].as_str() {
                Some(g) => g.to_string(),
                None => return RespValue::error("ERR invalid group name"),
            };

            let id = match args[3].as_str() {
                Some(id) => id,
                None => return RespValue::error("ERR invalid ID"),
            };

            let mkstream = args.get(4).and_then(|v| v.as_str()).map(|s| s.to_uppercase()) == Some("MKSTREAM".to_string());

            let mut streams = store.write().unwrap();

            if !streams.contains_key(&key) {
                if mkstream {
                    streams.insert(key.clone(), Stream::new());
                } else {
                    return RespValue::error("ERR The XGROUP subcommand requires the key to exist");
                }
            }

            let stream = streams.get_mut(&key).unwrap();
            match stream.create_group(groupname, id) {
                Ok(()) => RespValue::ok(),
                Err(e) => RespValue::error(e.to_string()),
            }
        }
        "DESTROY" => {
            if args.len() < 3 {
                return RespValue::error("ERR wrong number of arguments for 'xgroup destroy'");
            }

            let key = match args[1].as_str() {
                Some(k) => k,
                None => return RespValue::error("ERR invalid key"),
            };

            let groupname = match args[2].as_str() {
                Some(g) => g,
                None => return RespValue::error("ERR invalid group name"),
            };

            let mut streams = store.write().unwrap();
            match streams.get_mut(key) {
                Some(stream) => {
                    if stream.destroy_group(groupname) {
                        RespValue::integer(1)
                    } else {
                        RespValue::integer(0)
                    }
                }
                None => RespValue::integer(0),
            }
        }
        _ => RespValue::error(format!("ERR unknown subcommand '{}'", subcommand)),
    }
}

/// XREADGROUP GROUP group consumer [COUNT count] [BLOCK milliseconds] [NOACK] STREAMS key [key ...] id [id ...]
pub fn xreadgroup(args: &[RespValue], store: &StreamStore) -> RespValue {
    if args.len() < 5 {
        return RespValue::error("ERR wrong number of arguments for 'xreadgroup' command");
    }

    let mut idx = 0;
    let mut group: Option<String> = None;
    let mut consumer: Option<String> = None;
    let mut count: Option<usize> = None;
    let mut _noack = false;

    // Parse GROUP group consumer
    match args[idx].as_str().map(|s| s.to_uppercase()).as_deref() {
        Some("GROUP") => {
            idx += 1;
            group = args.get(idx).and_then(|v| v.as_str()).map(String::from);
            idx += 1;
            consumer = args.get(idx).and_then(|v| v.as_str()).map(String::from);
            idx += 1;
        }
        _ => return RespValue::error("ERR syntax error, expected GROUP"),
    }

    let group = match group {
        Some(g) => g,
        None => return RespValue::error("ERR invalid group name"),
    };

    let consumer = match consumer {
        Some(c) => c,
        None => return RespValue::error("ERR invalid consumer name"),
    };

    // Parse options
    while idx < args.len() {
        match args[idx].as_str().map(|s| s.to_uppercase()).as_deref() {
            Some("COUNT") => {
                idx += 1;
                count = args.get(idx).and_then(|v| v.as_int()).map(|c| c as usize);
                idx += 1;
            }
            Some("BLOCK") => {
                idx += 1;
                // Skip block value
                idx += 1;
            }
            Some("NOACK") => {
                _noack = true;
                idx += 1;
            }
            Some("STREAMS") => {
                idx += 1;
                break;
            }
            _ => idx += 1,
        }
    }

    // Parse keys and IDs
    let remaining = &args[idx..];
    if remaining.is_empty() || remaining.len() % 2 != 0 {
        return RespValue::error("ERR Unbalanced 'xreadgroup' list of streams");
    }

    let half = remaining.len() / 2;
    let keys: Vec<String> = remaining[..half]
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    let ids: Vec<&str> = remaining[half..]
        .iter()
        .filter_map(|v| v.as_str())
        .collect();

    if keys.len() != ids.len() {
        return RespValue::error("ERR Unbalanced 'xreadgroup' list of streams");
    }

    let mut streams = store.write().unwrap();
    let mut result = Vec::new();

    for (key, id_str) in keys.iter().zip(ids.iter()) {
        if let Some(stream) = streams.get_mut(key) {
            match stream.read_group(&group, &consumer, count, id_str) {
                Ok(entries) => {
                    if !entries.is_empty() {
                        let entry_values: Vec<RespValue> = entries
                            .iter()
                            .map(|entry| {
                                let fields: Vec<RespValue> = entry
                                    .fields
                                    .iter()
                                    .flat_map(|(k, v)| {
                                        vec![RespValue::bulk(k.clone()), RespValue::bulk(v.clone())]
                                    })
                                    .collect();
                                RespValue::array(vec![
                                    RespValue::bulk_string(entry.id.to_string()),
                                    RespValue::array(fields),
                                ])
                            })
                            .collect();

                        result.push(RespValue::array(vec![
                            RespValue::bulk_string(key.clone()),
                            RespValue::array(entry_values),
                        ]));
                    }
                }
                Err(e) => return RespValue::error(e.to_string()),
            }
        }
    }

    if result.is_empty() {
        RespValue::null()
    } else {
        RespValue::array(result)
    }
}

/// XACK key group id [id ...]
pub fn xack(args: &[RespValue], store: &StreamStore) -> RespValue {
    if args.len() < 3 {
        return RespValue::error("ERR wrong number of arguments for 'xack' command");
    }

    let key = match args[0].as_str() {
        Some(k) => k.to_string(),
        None => return RespValue::error("ERR invalid key"),
    };

    let group = match args[1].as_str() {
        Some(g) => g,
        None => return RespValue::error("ERR invalid group name"),
    };

    let ids: Result<Vec<StreamId>, _> = args[2..]
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or(StreamError::InvalidId)
                .and_then(|s| StreamId::parse(s, None))
        })
        .collect();

    let ids = match ids {
        Ok(ids) => ids,
        Err(e) => return RespValue::error(e.to_string()),
    };

    let mut streams = store.write().unwrap();
    match streams.get_mut(&key) {
        Some(stream) => match stream.ack(group, &ids) {
            Ok(count) => RespValue::integer(count as i64),
            Err(e) => RespValue::error(e.to_string()),
        },
        None => RespValue::integer(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_store() -> StreamStore {
        new_stream_store()
    }

    #[test]
    fn test_xadd() {
        let store = setup_store();
        let args = vec![
            RespValue::bulk_string("mystream"),
            RespValue::bulk_string("*"),
            RespValue::bulk_string("field1"),
            RespValue::bulk_string("value1"),
        ];

        let result = xadd(&args, &store);
        assert!(matches!(result, RespValue::BulkString(Some(_))));

        // Verify stream was created
        let streams = store.read().unwrap();
        assert!(streams.contains_key("mystream"));
        assert_eq!(streams.get("mystream").unwrap().len(), 1);
    }

    #[test]
    fn test_xadd_with_maxlen() {
        let store = setup_store();

        // Add 5 entries with MAXLEN 3
        for i in 0..5 {
            let args = vec![
                RespValue::bulk_string("mystream"),
                RespValue::bulk_string("MAXLEN"),
                RespValue::bulk_string("3"),
                RespValue::bulk_string("*"),
                RespValue::bulk_string("field"),
                RespValue::bulk_string(format!("value{}", i)),
            ];
            xadd(&args, &store);
        }

        let streams = store.read().unwrap();
        assert_eq!(streams.get("mystream").unwrap().len(), 3);
    }

    #[test]
    fn test_xlen() {
        let store = setup_store();

        // Empty stream
        let args = vec![RespValue::bulk_string("mystream")];
        assert_eq!(xlen(&args, &store), RespValue::integer(0));

        // Add some entries
        for _ in 0..3 {
            let add_args = vec![
                RespValue::bulk_string("mystream"),
                RespValue::bulk_string("*"),
                RespValue::bulk_string("f"),
                RespValue::bulk_string("v"),
            ];
            xadd(&add_args, &store);
        }

        assert_eq!(xlen(&args, &store), RespValue::integer(3));
    }

    #[test]
    fn test_xrange() {
        let store = setup_store();

        // Add entries
        for i in 0..5 {
            let args = vec![
                RespValue::bulk_string("mystream"),
                RespValue::bulk_string(format!("{}-0", 1000 + i)),
                RespValue::bulk_string("num"),
                RespValue::bulk_string(format!("{}", i)),
            ];
            xadd(&args, &store);
        }

        // Range query
        let args = vec![
            RespValue::bulk_string("mystream"),
            RespValue::bulk_string("-"),
            RespValue::bulk_string("+"),
        ];
        let result = xrange(&args, &store);
        match result {
            RespValue::Array(Some(arr)) => assert_eq!(arr.len(), 5),
            _ => panic!("Expected array"),
        }

        // Range with COUNT
        let args = vec![
            RespValue::bulk_string("mystream"),
            RespValue::bulk_string("-"),
            RespValue::bulk_string("+"),
            RespValue::bulk_string("COUNT"),
            RespValue::bulk_string("2"),
        ];
        let result = xrange(&args, &store);
        match result {
            RespValue::Array(Some(arr)) => assert_eq!(arr.len(), 2),
            _ => panic!("Expected array"),
        }
    }

    #[test]
    fn test_xread() {
        let store = setup_store();

        // Add entries
        for i in 0..3 {
            let args = vec![
                RespValue::bulk_string("mystream"),
                RespValue::bulk_string(format!("{}-0", 1000 + i)),
                RespValue::bulk_string("f"),
                RespValue::bulk_string("v"),
            ];
            xadd(&args, &store);
        }

        // Read from beginning
        let args = vec![
            RespValue::bulk_string("STREAMS"),
            RespValue::bulk_string("mystream"),
            RespValue::bulk_string("0"),
        ];
        let result = xread(&args, &store);
        assert!(!matches!(result, RespValue::BulkString(None)));
    }

    #[test]
    fn test_xgroup_create() {
        let store = setup_store();

        // Create stream first
        let args = vec![
            RespValue::bulk_string("mystream"),
            RespValue::bulk_string("*"),
            RespValue::bulk_string("f"),
            RespValue::bulk_string("v"),
        ];
        xadd(&args, &store);

        // Create group
        let args = vec![
            RespValue::bulk_string("CREATE"),
            RespValue::bulk_string("mystream"),
            RespValue::bulk_string("mygroup"),
            RespValue::bulk_string("0"),
        ];
        let result = xgroup(&args, &store);
        assert_eq!(result, RespValue::ok());
    }

    #[test]
    fn test_xtrim() {
        let store = setup_store();

        // Add entries
        for i in 0..10 {
            let args = vec![
                RespValue::bulk_string("mystream"),
                RespValue::bulk_string(format!("{}-0", 1000 + i)),
                RespValue::bulk_string("f"),
                RespValue::bulk_string("v"),
            ];
            xadd(&args, &store);
        }

        // Trim to 5
        let args = vec![
            RespValue::bulk_string("mystream"),
            RespValue::bulk_string("MAXLEN"),
            RespValue::bulk_string("5"),
        ];
        let result = xtrim(&args, &store);
        assert_eq!(result, RespValue::integer(5)); // Removed 5

        let len_args = vec![RespValue::bulk_string("mystream")];
        assert_eq!(xlen(&len_args, &store), RespValue::integer(5));
    }

    #[test]
    fn test_xdel() {
        let store = setup_store();

        // Add entries
        for i in 0..5 {
            let args = vec![
                RespValue::bulk_string("mystream"),
                RespValue::bulk_string(format!("{}-0", 1000 + i)),
                RespValue::bulk_string("f"),
                RespValue::bulk_string("v"),
            ];
            xadd(&args, &store);
        }

        // Delete specific entry
        let args = vec![
            RespValue::bulk_string("mystream"),
            RespValue::bulk_string("1002-0"),
        ];
        let result = xdel(&args, &store);
        assert_eq!(result, RespValue::integer(1));

        let len_args = vec![RespValue::bulk_string("mystream")];
        assert_eq!(xlen(&len_args, &store), RespValue::integer(4));
    }

    #[test]
    fn test_xreadgroup_and_xack() {
        let store = setup_store();

        // Add entries
        for i in 0..3 {
            let args = vec![
                RespValue::bulk_string("mystream"),
                RespValue::bulk_string(format!("{}-0", 1000 + i)),
                RespValue::bulk_string("f"),
                RespValue::bulk_string("v"),
            ];
            xadd(&args, &store);
        }

        // Create group
        let args = vec![
            RespValue::bulk_string("CREATE"),
            RespValue::bulk_string("mystream"),
            RespValue::bulk_string("mygroup"),
            RespValue::bulk_string("0"),
        ];
        xgroup(&args, &store);

        // Read from group
        let args = vec![
            RespValue::bulk_string("GROUP"),
            RespValue::bulk_string("mygroup"),
            RespValue::bulk_string("consumer1"),
            RespValue::bulk_string("COUNT"),
            RespValue::bulk_string("1"),
            RespValue::bulk_string("STREAMS"),
            RespValue::bulk_string("mystream"),
            RespValue::bulk_string(">"),
        ];
        let result = xreadgroup(&args, &store);
        assert!(!matches!(result, RespValue::BulkString(None)));

        // ACK the entry
        let args = vec![
            RespValue::bulk_string("mystream"),
            RespValue::bulk_string("mygroup"),
            RespValue::bulk_string("1000-0"),
        ];
        let result = xack(&args, &store);
        assert_eq!(result, RespValue::integer(1));
    }
}
