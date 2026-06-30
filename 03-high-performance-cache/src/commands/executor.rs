use crate::resp::RespValue;
use crate::storage::Database;

use super::keys;
use super::server_cmd;
use super::strings;
use super::lists;
use super::sets;
use super::hashes;
use super::zsets;
use super::pubsub_cmd;
use super::transaction_cmd;
use super::cluster_cmd;

/// Command executor
pub struct CommandExecutor;

impl CommandExecutor {
    /// Execute a command
    pub fn execute(cmd: &str, args: &[RespValue], db: &mut Database) -> RespValue {
        match cmd {
            // String commands
            "GET" => strings::get(args, db),
            "SET" => strings::set(args, db),
            "SETNX" => strings::setnx(args, db),
            "SETEX" => strings::setex(args, db),
            "PSETEX" => strings::psetex(args, db),
            "MGET" => strings::mget(args, db),
            "MSET" => strings::mset(args, db),
            "APPEND" => strings::append(args, db),
            "STRLEN" => strings::strlen(args, db),
            "INCR" => strings::incr(args, db),
            "INCRBY" => strings::incrby(args, db),
            "DECR" => strings::decr(args, db),
            "DECRBY" => strings::decrby(args, db),
            "GETSET" => strings::getset(args, db),

            // List commands
            "LPUSH" => lists::lpush(args, db),
            "RPUSH" => lists::rpush(args, db),
            "LPOP" => lists::lpop(args, db),
            "RPOP" => lists::rpop(args, db),
            "LLEN" => lists::llen(args, db),
            "LRANGE" => lists::lrange(args, db),
            "LINDEX" => lists::lindex(args, db),
            "LSET" => lists::lset(args, db),

            // Set commands
            "SADD" => sets::sadd(args, db),
            "SREM" => sets::srem(args, db),
            "SMEMBERS" => sets::smembers(args, db),
            "SISMEMBER" => sets::sismember(args, db),
            "SCARD" => sets::scard(args, db),
            "SINTER" => sets::sinter(args, db),
            "SUNION" => sets::sunion(args, db),
            "SDIFF" => sets::sdiff(args, db),

            // Hash commands
            "HSET" => hashes::hset(args, db),
            "HGET" => hashes::hget(args, db),
            "HDEL" => hashes::hdel(args, db),
            "HEXISTS" => hashes::hexists(args, db),
            "HLEN" => hashes::hlen(args, db),
            "HGETALL" => hashes::hgetall(args, db),
            "HKEYS" => hashes::hkeys(args, db),
            "HVALS" => hashes::hvals(args, db),
            "HMSET" => hashes::hmset(args, db),
            "HMGET" => hashes::hmget(args, db),

            // Sorted set commands
            "ZADD" => zsets::zadd(args, db),
            "ZREM" => zsets::zrem(args, db),
            "ZSCORE" => zsets::zscore(args, db),
            "ZCARD" => zsets::zcard(args, db),
            "ZRANGE" => zsets::zrange(args, db),
            "ZRANK" => zsets::zrank(args, db),
            "ZCOUNT" => zsets::zcount(args, db),
            "ZINCRBY" => zsets::zincrby(args, db),

            // Key commands
            "DEL" => keys::del(args, db),
            "EXISTS" => keys::exists(args, db),
            "EXPIRE" => keys::expire(args, db),
            "EXPIREAT" => keys::expireat(args, db),
            "PEXPIRE" => keys::pexpire(args, db),
            "TTL" => keys::ttl(args, db),
            "PTTL" => keys::pttl(args, db),
            "PERSIST" => keys::persist(args, db),
            "TYPE" => keys::key_type(args, db),
            "KEYS" => keys::keys(args, db),
            "DBSIZE" => keys::dbsize(args, db),
            "FLUSHDB" => keys::flushdb(args, db),
            "RENAME" => keys::rename(args, db),
            "RENAMENX" => keys::renamenx(args, db),

            // Server commands
            "PING" => server_cmd::ping(args),
            "ECHO" => server_cmd::echo(args),
            "INFO" => server_cmd::info(args, db),
            "COMMAND" => server_cmd::command(args),

            // Pub/Sub commands
            "PUBLISH" => pubsub_cmd::publish(args, db),
            "PUBSUB" => pubsub_cmd::pubsub(args, db),

            // Transaction commands
            "MULTI" => transaction_cmd::multi(args, db),
            "EXEC" => transaction_cmd::exec(args, db),
            "DISCARD" => transaction_cmd::discard(args, db),
            "WATCH" => transaction_cmd::watch(args, db),
            "UNWATCH" => transaction_cmd::unwatch(args, db),

            // Cluster commands
            "CLUSTER" => cluster_cmd::cluster(args, None),
            "READONLY" => cluster_cmd::readonly(args),
            "READWRITE" => cluster_cmd::readwrite(args),

            _ => RespValue::error(format!("ERR unknown command '{}'", cmd)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== String Command Tests ====================

    #[test]
    fn test_get_set() {
        let mut db = Database::new();

        // SET key value
        let result = CommandExecutor::execute(
            "SET",
            &[RespValue::bulk_string("foo"), RespValue::bulk_string("bar")],
            &mut db,
        );
        assert_eq!(result, RespValue::ok());

        // GET key
        let result = CommandExecutor::execute(
            "GET",
            &[RespValue::bulk_string("foo")],
            &mut db,
        );
        assert_eq!(result, RespValue::BulkString(Some(b"bar".to_vec())));
    }

    #[test]
    fn test_get_nonexistent() {
        let mut db = Database::new();
        let result = CommandExecutor::execute(
            "GET",
            &[RespValue::bulk_string("nonexistent")],
            &mut db,
        );
        assert_eq!(result, RespValue::null());
    }

    #[test]
    fn test_set_with_ex() {
        let mut db = Database::new();
        let result = CommandExecutor::execute(
            "SET",
            &[
                RespValue::bulk_string("key"),
                RespValue::bulk_string("value"),
                RespValue::bulk_string("EX"),
                RespValue::bulk_string("60"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::ok());
        assert!(db.exists("key"));
    }

    #[test]
    fn test_set_nx() {
        let mut db = Database::new();

        // SET key value NX - should succeed
        let result = CommandExecutor::execute(
            "SET",
            &[
                RespValue::bulk_string("key"),
                RespValue::bulk_string("value1"),
                RespValue::bulk_string("NX"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::ok());

        // SET key value NX - should fail (key exists)
        let result = CommandExecutor::execute(
            "SET",
            &[
                RespValue::bulk_string("key"),
                RespValue::bulk_string("value2"),
                RespValue::bulk_string("NX"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::null());

        // Value should not have changed
        let result = CommandExecutor::execute(
            "GET",
            &[RespValue::bulk_string("key")],
            &mut db,
        );
        assert_eq!(result, RespValue::BulkString(Some(b"value1".to_vec())));
    }

    #[test]
    fn test_set_xx() {
        let mut db = Database::new();

        // SET key value XX - should fail (key doesn't exist)
        let result = CommandExecutor::execute(
            "SET",
            &[
                RespValue::bulk_string("key"),
                RespValue::bulk_string("value1"),
                RespValue::bulk_string("XX"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::null());

        // Create the key first
        CommandExecutor::execute(
            "SET",
            &[RespValue::bulk_string("key"), RespValue::bulk_string("original")],
            &mut db,
        );

        // SET key value XX - should succeed (key exists)
        let result = CommandExecutor::execute(
            "SET",
            &[
                RespValue::bulk_string("key"),
                RespValue::bulk_string("value2"),
                RespValue::bulk_string("XX"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::ok());
    }

    #[test]
    fn test_setnx() {
        let mut db = Database::new();

        let result = CommandExecutor::execute(
            "SETNX",
            &[RespValue::bulk_string("key"), RespValue::bulk_string("value")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(1));

        let result = CommandExecutor::execute(
            "SETNX",
            &[RespValue::bulk_string("key"), RespValue::bulk_string("other")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(0));
    }

    #[test]
    fn test_mget_mset() {
        let mut db = Database::new();

        // MSET
        let result = CommandExecutor::execute(
            "MSET",
            &[
                RespValue::bulk_string("k1"),
                RespValue::bulk_string("v1"),
                RespValue::bulk_string("k2"),
                RespValue::bulk_string("v2"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::ok());

        // MGET
        let result = CommandExecutor::execute(
            "MGET",
            &[
                RespValue::bulk_string("k1"),
                RespValue::bulk_string("k2"),
                RespValue::bulk_string("k3"),
            ],
            &mut db,
        );
        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0], RespValue::BulkString(Some(b"v1".to_vec())));
                assert_eq!(arr[1], RespValue::BulkString(Some(b"v2".to_vec())));
                assert_eq!(arr[2], RespValue::null());
            }
            _ => panic!("Expected array"),
        }
    }

    #[test]
    fn test_incr_decr() {
        let mut db = Database::new();

        // INCR on new key
        let result = CommandExecutor::execute(
            "INCR",
            &[RespValue::bulk_string("counter")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(1));

        // INCR again
        let result = CommandExecutor::execute(
            "INCR",
            &[RespValue::bulk_string("counter")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(2));

        // INCRBY
        let result = CommandExecutor::execute(
            "INCRBY",
            &[RespValue::bulk_string("counter"), RespValue::bulk_string("10")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(12));

        // DECR
        let result = CommandExecutor::execute(
            "DECR",
            &[RespValue::bulk_string("counter")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(11));

        // DECRBY
        let result = CommandExecutor::execute(
            "DECRBY",
            &[RespValue::bulk_string("counter"), RespValue::bulk_string("5")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(6));
    }

    #[test]
    fn test_append_strlen() {
        let mut db = Database::new();

        // APPEND to new key
        let result = CommandExecutor::execute(
            "APPEND",
            &[RespValue::bulk_string("key"), RespValue::bulk_string("hello")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(5));

        // APPEND to existing key
        let result = CommandExecutor::execute(
            "APPEND",
            &[RespValue::bulk_string("key"), RespValue::bulk_string(" world")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(11));

        // STRLEN
        let result = CommandExecutor::execute(
            "STRLEN",
            &[RespValue::bulk_string("key")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(11));
    }

    #[test]
    fn test_getset() {
        let mut db = Database::new();

        // GETSET on new key
        let result = CommandExecutor::execute(
            "GETSET",
            &[RespValue::bulk_string("key"), RespValue::bulk_string("value1")],
            &mut db,
        );
        assert_eq!(result, RespValue::null());

        // GETSET on existing key
        let result = CommandExecutor::execute(
            "GETSET",
            &[RespValue::bulk_string("key"), RespValue::bulk_string("value2")],
            &mut db,
        );
        assert_eq!(result, RespValue::BulkString(Some(b"value1".to_vec())));
    }

    // ==================== List Command Tests ====================

    #[test]
    fn test_lpush_rpush() {
        let mut db = Database::new();

        // LPUSH
        let result = CommandExecutor::execute(
            "LPUSH",
            &[
                RespValue::bulk_string("list"),
                RespValue::bulk_string("a"),
                RespValue::bulk_string("b"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(2));

        // RPUSH
        let result = CommandExecutor::execute(
            "RPUSH",
            &[
                RespValue::bulk_string("list"),
                RespValue::bulk_string("c"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(3));

        // LLEN
        let result = CommandExecutor::execute(
            "LLEN",
            &[RespValue::bulk_string("list")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(3));
    }

    #[test]
    fn test_lpop_rpop() {
        let mut db = Database::new();

        // Setup list
        CommandExecutor::execute(
            "RPUSH",
            &[
                RespValue::bulk_string("list"),
                RespValue::bulk_string("a"),
                RespValue::bulk_string("b"),
                RespValue::bulk_string("c"),
            ],
            &mut db,
        );

        // LPOP
        let result = CommandExecutor::execute(
            "LPOP",
            &[RespValue::bulk_string("list")],
            &mut db,
        );
        assert_eq!(result, RespValue::BulkString(Some(b"a".to_vec())));

        // RPOP
        let result = CommandExecutor::execute(
            "RPOP",
            &[RespValue::bulk_string("list")],
            &mut db,
        );
        assert_eq!(result, RespValue::BulkString(Some(b"c".to_vec())));
    }

    #[test]
    fn test_lrange() {
        let mut db = Database::new();

        // Setup list
        CommandExecutor::execute(
            "RPUSH",
            &[
                RespValue::bulk_string("list"),
                RespValue::bulk_string("a"),
                RespValue::bulk_string("b"),
                RespValue::bulk_string("c"),
                RespValue::bulk_string("d"),
            ],
            &mut db,
        );

        // LRANGE 0 -1 (entire list)
        let result = CommandExecutor::execute(
            "LRANGE",
            &[
                RespValue::bulk_string("list"),
                RespValue::bulk_string("0"),
                RespValue::bulk_string("-1"),
            ],
            &mut db,
        );
        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 4);
            }
            _ => panic!("Expected array"),
        }

        // LRANGE 1 2
        let result = CommandExecutor::execute(
            "LRANGE",
            &[
                RespValue::bulk_string("list"),
                RespValue::bulk_string("1"),
                RespValue::bulk_string("2"),
            ],
            &mut db,
        );
        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 2);
                assert_eq!(arr[0], RespValue::BulkString(Some(b"b".to_vec())));
                assert_eq!(arr[1], RespValue::BulkString(Some(b"c".to_vec())));
            }
            _ => panic!("Expected array"),
        }
    }

    #[test]
    fn test_lindex_lset() {
        let mut db = Database::new();

        // Setup list
        CommandExecutor::execute(
            "RPUSH",
            &[
                RespValue::bulk_string("list"),
                RespValue::bulk_string("a"),
                RespValue::bulk_string("b"),
                RespValue::bulk_string("c"),
            ],
            &mut db,
        );

        // LINDEX
        let result = CommandExecutor::execute(
            "LINDEX",
            &[RespValue::bulk_string("list"), RespValue::bulk_string("1")],
            &mut db,
        );
        assert_eq!(result, RespValue::BulkString(Some(b"b".to_vec())));

        // LINDEX with negative index
        let result = CommandExecutor::execute(
            "LINDEX",
            &[RespValue::bulk_string("list"), RespValue::bulk_string("-1")],
            &mut db,
        );
        assert_eq!(result, RespValue::BulkString(Some(b"c".to_vec())));

        // LSET
        let result = CommandExecutor::execute(
            "LSET",
            &[
                RespValue::bulk_string("list"),
                RespValue::bulk_string("1"),
                RespValue::bulk_string("B"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::ok());

        // Verify LSET worked
        let result = CommandExecutor::execute(
            "LINDEX",
            &[RespValue::bulk_string("list"), RespValue::bulk_string("1")],
            &mut db,
        );
        assert_eq!(result, RespValue::BulkString(Some(b"B".to_vec())));
    }

    // ==================== Set Command Tests ====================

    #[test]
    fn test_sadd_srem_sismember() {
        let mut db = Database::new();

        // SADD
        let result = CommandExecutor::execute(
            "SADD",
            &[
                RespValue::bulk_string("set"),
                RespValue::bulk_string("a"),
                RespValue::bulk_string("b"),
                RespValue::bulk_string("c"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(3));

        // SADD duplicates
        let result = CommandExecutor::execute(
            "SADD",
            &[
                RespValue::bulk_string("set"),
                RespValue::bulk_string("a"),
                RespValue::bulk_string("d"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(1));

        // SISMEMBER
        let result = CommandExecutor::execute(
            "SISMEMBER",
            &[RespValue::bulk_string("set"), RespValue::bulk_string("a")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(1));

        let result = CommandExecutor::execute(
            "SISMEMBER",
            &[RespValue::bulk_string("set"), RespValue::bulk_string("x")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(0));

        // SREM
        let result = CommandExecutor::execute(
            "SREM",
            &[
                RespValue::bulk_string("set"),
                RespValue::bulk_string("a"),
                RespValue::bulk_string("x"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(1));

        // SCARD
        let result = CommandExecutor::execute(
            "SCARD",
            &[RespValue::bulk_string("set")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(3)); // b, c, d
    }

    #[test]
    fn test_smembers() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "SADD",
            &[
                RespValue::bulk_string("set"),
                RespValue::bulk_string("a"),
                RespValue::bulk_string("b"),
            ],
            &mut db,
        );

        let result = CommandExecutor::execute(
            "SMEMBERS",
            &[RespValue::bulk_string("set")],
            &mut db,
        );
        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 2);
            }
            _ => panic!("Expected array"),
        }
    }

    #[test]
    fn test_sinter_sunion_sdiff() {
        let mut db = Database::new();

        // Setup sets
        CommandExecutor::execute(
            "SADD",
            &[
                RespValue::bulk_string("set1"),
                RespValue::bulk_string("a"),
                RespValue::bulk_string("b"),
                RespValue::bulk_string("c"),
            ],
            &mut db,
        );
        CommandExecutor::execute(
            "SADD",
            &[
                RespValue::bulk_string("set2"),
                RespValue::bulk_string("b"),
                RespValue::bulk_string("c"),
                RespValue::bulk_string("d"),
            ],
            &mut db,
        );

        // SINTER
        let result = CommandExecutor::execute(
            "SINTER",
            &[RespValue::bulk_string("set1"), RespValue::bulk_string("set2")],
            &mut db,
        );
        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 2); // b, c
            }
            _ => panic!("Expected array"),
        }

        // SUNION
        let result = CommandExecutor::execute(
            "SUNION",
            &[RespValue::bulk_string("set1"), RespValue::bulk_string("set2")],
            &mut db,
        );
        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 4); // a, b, c, d
            }
            _ => panic!("Expected array"),
        }

        // SDIFF
        let result = CommandExecutor::execute(
            "SDIFF",
            &[RespValue::bulk_string("set1"), RespValue::bulk_string("set2")],
            &mut db,
        );
        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 1); // a
            }
            _ => panic!("Expected array"),
        }
    }

    // ==================== Hash Command Tests ====================

    #[test]
    fn test_hset_hget() {
        let mut db = Database::new();

        // HSET
        let result = CommandExecutor::execute(
            "HSET",
            &[
                RespValue::bulk_string("hash"),
                RespValue::bulk_string("field1"),
                RespValue::bulk_string("value1"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(1));

        // HSET multiple fields
        let result = CommandExecutor::execute(
            "HSET",
            &[
                RespValue::bulk_string("hash"),
                RespValue::bulk_string("field2"),
                RespValue::bulk_string("value2"),
                RespValue::bulk_string("field3"),
                RespValue::bulk_string("value3"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(2));

        // HGET
        let result = CommandExecutor::execute(
            "HGET",
            &[RespValue::bulk_string("hash"), RespValue::bulk_string("field1")],
            &mut db,
        );
        assert_eq!(result, RespValue::BulkString(Some(b"value1".to_vec())));

        // HGET nonexistent field
        let result = CommandExecutor::execute(
            "HGET",
            &[RespValue::bulk_string("hash"), RespValue::bulk_string("nonexistent")],
            &mut db,
        );
        assert_eq!(result, RespValue::null());
    }

    #[test]
    fn test_hdel_hexists() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "HSET",
            &[
                RespValue::bulk_string("hash"),
                RespValue::bulk_string("f1"),
                RespValue::bulk_string("v1"),
                RespValue::bulk_string("f2"),
                RespValue::bulk_string("v2"),
            ],
            &mut db,
        );

        // HEXISTS
        let result = CommandExecutor::execute(
            "HEXISTS",
            &[RespValue::bulk_string("hash"), RespValue::bulk_string("f1")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(1));

        // HDEL
        let result = CommandExecutor::execute(
            "HDEL",
            &[RespValue::bulk_string("hash"), RespValue::bulk_string("f1")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(1));

        // HEXISTS after delete
        let result = CommandExecutor::execute(
            "HEXISTS",
            &[RespValue::bulk_string("hash"), RespValue::bulk_string("f1")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(0));
    }

    #[test]
    fn test_hlen_hgetall() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "HSET",
            &[
                RespValue::bulk_string("hash"),
                RespValue::bulk_string("f1"),
                RespValue::bulk_string("v1"),
                RespValue::bulk_string("f2"),
                RespValue::bulk_string("v2"),
            ],
            &mut db,
        );

        // HLEN
        let result = CommandExecutor::execute(
            "HLEN",
            &[RespValue::bulk_string("hash")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(2));

        // HGETALL
        let result = CommandExecutor::execute(
            "HGETALL",
            &[RespValue::bulk_string("hash")],
            &mut db,
        );
        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 4); // 2 fields * 2 (field + value)
            }
            _ => panic!("Expected array"),
        }
    }

    #[test]
    fn test_hkeys_hvals() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "HSET",
            &[
                RespValue::bulk_string("hash"),
                RespValue::bulk_string("f1"),
                RespValue::bulk_string("v1"),
                RespValue::bulk_string("f2"),
                RespValue::bulk_string("v2"),
            ],
            &mut db,
        );

        // HKEYS
        let result = CommandExecutor::execute(
            "HKEYS",
            &[RespValue::bulk_string("hash")],
            &mut db,
        );
        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 2);
            }
            _ => panic!("Expected array"),
        }

        // HVALS
        let result = CommandExecutor::execute(
            "HVALS",
            &[RespValue::bulk_string("hash")],
            &mut db,
        );
        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 2);
            }
            _ => panic!("Expected array"),
        }
    }

    #[test]
    fn test_hmset_hmget() {
        let mut db = Database::new();

        // HMSET
        let result = CommandExecutor::execute(
            "HMSET",
            &[
                RespValue::bulk_string("hash"),
                RespValue::bulk_string("f1"),
                RespValue::bulk_string("v1"),
                RespValue::bulk_string("f2"),
                RespValue::bulk_string("v2"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::ok());

        // HMGET
        let result = CommandExecutor::execute(
            "HMGET",
            &[
                RespValue::bulk_string("hash"),
                RespValue::bulk_string("f1"),
                RespValue::bulk_string("f2"),
                RespValue::bulk_string("f3"),
            ],
            &mut db,
        );
        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0], RespValue::BulkString(Some(b"v1".to_vec())));
                assert_eq!(arr[1], RespValue::BulkString(Some(b"v2".to_vec())));
                assert_eq!(arr[2], RespValue::null());
            }
            _ => panic!("Expected array"),
        }
    }

    // ==================== Sorted Set Command Tests ====================

    #[test]
    fn test_zadd_zscore() {
        let mut db = Database::new();

        // ZADD
        let result = CommandExecutor::execute(
            "ZADD",
            &[
                RespValue::bulk_string("zset"),
                RespValue::bulk_string("1.0"),
                RespValue::bulk_string("one"),
                RespValue::bulk_string("2.0"),
                RespValue::bulk_string("two"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(2));

        // ZSCORE
        let result = CommandExecutor::execute(
            "ZSCORE",
            &[RespValue::bulk_string("zset"), RespValue::bulk_string("one")],
            &mut db,
        );
        assert_eq!(result, RespValue::BulkString(Some(b"1".to_vec())));

        // ZSCORE nonexistent
        let result = CommandExecutor::execute(
            "ZSCORE",
            &[RespValue::bulk_string("zset"), RespValue::bulk_string("three")],
            &mut db,
        );
        assert_eq!(result, RespValue::null());
    }

    #[test]
    fn test_zrem_zcard() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "ZADD",
            &[
                RespValue::bulk_string("zset"),
                RespValue::bulk_string("1"),
                RespValue::bulk_string("a"),
                RespValue::bulk_string("2"),
                RespValue::bulk_string("b"),
                RespValue::bulk_string("3"),
                RespValue::bulk_string("c"),
            ],
            &mut db,
        );

        // ZCARD
        let result = CommandExecutor::execute(
            "ZCARD",
            &[RespValue::bulk_string("zset")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(3));

        // ZREM
        let result = CommandExecutor::execute(
            "ZREM",
            &[RespValue::bulk_string("zset"), RespValue::bulk_string("b")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(1));

        // ZCARD after ZREM
        let result = CommandExecutor::execute(
            "ZCARD",
            &[RespValue::bulk_string("zset")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(2));
    }

    #[test]
    fn test_zrange_zrank() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "ZADD",
            &[
                RespValue::bulk_string("zset"),
                RespValue::bulk_string("1"),
                RespValue::bulk_string("a"),
                RespValue::bulk_string("2"),
                RespValue::bulk_string("b"),
                RespValue::bulk_string("3"),
                RespValue::bulk_string("c"),
            ],
            &mut db,
        );

        // ZRANGE
        let result = CommandExecutor::execute(
            "ZRANGE",
            &[
                RespValue::bulk_string("zset"),
                RespValue::bulk_string("0"),
                RespValue::bulk_string("-1"),
            ],
            &mut db,
        );
        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0], RespValue::BulkString(Some(b"a".to_vec())));
                assert_eq!(arr[1], RespValue::BulkString(Some(b"b".to_vec())));
                assert_eq!(arr[2], RespValue::BulkString(Some(b"c".to_vec())));
            }
            _ => panic!("Expected array"),
        }

        // ZRANK
        let result = CommandExecutor::execute(
            "ZRANK",
            &[RespValue::bulk_string("zset"), RespValue::bulk_string("b")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(1));
    }

    #[test]
    fn test_zrange_withscores() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "ZADD",
            &[
                RespValue::bulk_string("zset"),
                RespValue::bulk_string("1"),
                RespValue::bulk_string("a"),
                RespValue::bulk_string("2"),
                RespValue::bulk_string("b"),
            ],
            &mut db,
        );

        // ZRANGE WITHSCORES
        let result = CommandExecutor::execute(
            "ZRANGE",
            &[
                RespValue::bulk_string("zset"),
                RespValue::bulk_string("0"),
                RespValue::bulk_string("-1"),
                RespValue::bulk_string("WITHSCORES"),
            ],
            &mut db,
        );
        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 4); // 2 members * 2 (member + score)
            }
            _ => panic!("Expected array"),
        }
    }

    #[test]
    fn test_zcount() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "ZADD",
            &[
                RespValue::bulk_string("zset"),
                RespValue::bulk_string("1"),
                RespValue::bulk_string("a"),
                RespValue::bulk_string("2"),
                RespValue::bulk_string("b"),
                RespValue::bulk_string("3"),
                RespValue::bulk_string("c"),
                RespValue::bulk_string("4"),
                RespValue::bulk_string("d"),
            ],
            &mut db,
        );

        // ZCOUNT 2 3
        let result = CommandExecutor::execute(
            "ZCOUNT",
            &[
                RespValue::bulk_string("zset"),
                RespValue::bulk_string("2"),
                RespValue::bulk_string("3"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(2));

        // ZCOUNT -inf +inf
        let result = CommandExecutor::execute(
            "ZCOUNT",
            &[
                RespValue::bulk_string("zset"),
                RespValue::bulk_string("-inf"),
                RespValue::bulk_string("+inf"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(4));
    }

    #[test]
    fn test_zincrby() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "ZADD",
            &[
                RespValue::bulk_string("zset"),
                RespValue::bulk_string("5"),
                RespValue::bulk_string("member"),
            ],
            &mut db,
        );

        // ZINCRBY
        let result = CommandExecutor::execute(
            "ZINCRBY",
            &[
                RespValue::bulk_string("zset"),
                RespValue::bulk_string("2.5"),
                RespValue::bulk_string("member"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::BulkString(Some(b"7.5".to_vec())));
    }

    // ==================== Key Command Tests ====================

    #[test]
    fn test_del_exists() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "SET",
            &[RespValue::bulk_string("key1"), RespValue::bulk_string("value1")],
            &mut db,
        );
        CommandExecutor::execute(
            "SET",
            &[RespValue::bulk_string("key2"), RespValue::bulk_string("value2")],
            &mut db,
        );

        // EXISTS
        let result = CommandExecutor::execute(
            "EXISTS",
            &[RespValue::bulk_string("key1")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(1));

        // DEL
        let result = CommandExecutor::execute(
            "DEL",
            &[
                RespValue::bulk_string("key1"),
                RespValue::bulk_string("key2"),
                RespValue::bulk_string("key3"),
            ],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(2));

        // EXISTS after DEL
        let result = CommandExecutor::execute(
            "EXISTS",
            &[RespValue::bulk_string("key1")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(0));
    }

    #[test]
    fn test_type() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "SET",
            &[RespValue::bulk_string("str"), RespValue::bulk_string("value")],
            &mut db,
        );
        CommandExecutor::execute(
            "LPUSH",
            &[RespValue::bulk_string("list"), RespValue::bulk_string("value")],
            &mut db,
        );
        CommandExecutor::execute(
            "SADD",
            &[RespValue::bulk_string("set"), RespValue::bulk_string("value")],
            &mut db,
        );
        CommandExecutor::execute(
            "HSET",
            &[
                RespValue::bulk_string("hash"),
                RespValue::bulk_string("field"),
                RespValue::bulk_string("value"),
            ],
            &mut db,
        );
        CommandExecutor::execute(
            "ZADD",
            &[
                RespValue::bulk_string("zset"),
                RespValue::bulk_string("1"),
                RespValue::bulk_string("value"),
            ],
            &mut db,
        );

        // TYPE string
        let result = CommandExecutor::execute(
            "TYPE",
            &[RespValue::bulk_string("str")],
            &mut db,
        );
        assert_eq!(result, RespValue::SimpleString("string".to_string()));

        // TYPE list
        let result = CommandExecutor::execute(
            "TYPE",
            &[RespValue::bulk_string("list")],
            &mut db,
        );
        assert_eq!(result, RespValue::SimpleString("list".to_string()));

        // TYPE set
        let result = CommandExecutor::execute(
            "TYPE",
            &[RespValue::bulk_string("set")],
            &mut db,
        );
        assert_eq!(result, RespValue::SimpleString("set".to_string()));

        // TYPE hash
        let result = CommandExecutor::execute(
            "TYPE",
            &[RespValue::bulk_string("hash")],
            &mut db,
        );
        assert_eq!(result, RespValue::SimpleString("hash".to_string()));

        // TYPE zset
        let result = CommandExecutor::execute(
            "TYPE",
            &[RespValue::bulk_string("zset")],
            &mut db,
        );
        assert_eq!(result, RespValue::SimpleString("zset".to_string()));

        // TYPE nonexistent
        let result = CommandExecutor::execute(
            "TYPE",
            &[RespValue::bulk_string("nonexistent")],
            &mut db,
        );
        assert_eq!(result, RespValue::SimpleString("none".to_string()));
    }

    #[test]
    fn test_dbsize_flushdb() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "SET",
            &[RespValue::bulk_string("k1"), RespValue::bulk_string("v1")],
            &mut db,
        );
        CommandExecutor::execute(
            "SET",
            &[RespValue::bulk_string("k2"), RespValue::bulk_string("v2")],
            &mut db,
        );

        // DBSIZE
        let result = CommandExecutor::execute("DBSIZE", &[], &mut db);
        assert_eq!(result, RespValue::Integer(2));

        // FLUSHDB
        let result = CommandExecutor::execute("FLUSHDB", &[], &mut db);
        assert_eq!(result, RespValue::ok());

        // DBSIZE after FLUSHDB
        let result = CommandExecutor::execute("DBSIZE", &[], &mut db);
        assert_eq!(result, RespValue::Integer(0));
    }

    #[test]
    fn test_rename() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "SET",
            &[RespValue::bulk_string("oldkey"), RespValue::bulk_string("value")],
            &mut db,
        );

        // RENAME
        let result = CommandExecutor::execute(
            "RENAME",
            &[RespValue::bulk_string("oldkey"), RespValue::bulk_string("newkey")],
            &mut db,
        );
        assert_eq!(result, RespValue::ok());

        // Verify old key is gone
        let result = CommandExecutor::execute(
            "EXISTS",
            &[RespValue::bulk_string("oldkey")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(0));

        // Verify new key exists
        let result = CommandExecutor::execute(
            "GET",
            &[RespValue::bulk_string("newkey")],
            &mut db,
        );
        assert_eq!(result, RespValue::BulkString(Some(b"value".to_vec())));
    }

    #[test]
    fn test_renamenx() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "SET",
            &[RespValue::bulk_string("key1"), RespValue::bulk_string("value1")],
            &mut db,
        );
        CommandExecutor::execute(
            "SET",
            &[RespValue::bulk_string("key2"), RespValue::bulk_string("value2")],
            &mut db,
        );

        // RENAMENX should fail (key2 exists)
        let result = CommandExecutor::execute(
            "RENAMENX",
            &[RespValue::bulk_string("key1"), RespValue::bulk_string("key2")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(0));

        // RENAMENX should succeed (key3 doesn't exist)
        let result = CommandExecutor::execute(
            "RENAMENX",
            &[RespValue::bulk_string("key1"), RespValue::bulk_string("key3")],
            &mut db,
        );
        assert_eq!(result, RespValue::Integer(1));
    }

    // ==================== Server Command Tests ====================

    #[test]
    fn test_ping() {
        let mut db = Database::new();

        let result = CommandExecutor::execute("PING", &[], &mut db);
        assert_eq!(result, RespValue::SimpleString("PONG".to_string()));

        let result = CommandExecutor::execute(
            "PING",
            &[RespValue::bulk_string("hello")],
            &mut db,
        );
        assert_eq!(result, RespValue::BulkString(Some(b"hello".to_vec())));
    }

    #[test]
    fn test_echo() {
        let mut db = Database::new();

        let result = CommandExecutor::execute(
            "ECHO",
            &[RespValue::bulk_string("hello world")],
            &mut db,
        );
        assert_eq!(result, RespValue::BulkString(Some(b"hello world".to_vec())));
    }

    #[test]
    fn test_unknown_command() {
        let mut db = Database::new();

        let result = CommandExecutor::execute("UNKNOWNCMD", &[], &mut db);
        match result {
            RespValue::Error(msg) => {
                assert!(msg.contains("unknown command"));
            }
            _ => panic!("Expected error"),
        }
    }

    // ==================== WRONGTYPE Error Tests ====================

    #[test]
    fn test_wrongtype_string_on_list() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "LPUSH",
            &[RespValue::bulk_string("list"), RespValue::bulk_string("value")],
            &mut db,
        );

        let result = CommandExecutor::execute(
            "GET",
            &[RespValue::bulk_string("list")],
            &mut db,
        );
        // GET on a list should return null (type mismatch handled by get_string)
        assert_eq!(result, RespValue::null());
    }

    #[test]
    fn test_wrongtype_list_on_string() {
        let mut db = Database::new();

        CommandExecutor::execute(
            "SET",
            &[RespValue::bulk_string("str"), RespValue::bulk_string("value")],
            &mut db,
        );

        let result = CommandExecutor::execute(
            "LPUSH",
            &[RespValue::bulk_string("str"), RespValue::bulk_string("value")],
            &mut db,
        );
        match result {
            RespValue::Error(msg) => {
                assert!(msg.contains("WRONGTYPE"));
            }
            _ => panic!("Expected WRONGTYPE error"),
        }
    }
}
