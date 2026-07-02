use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use crate::commands::CommandExecutor;
use crate::resp::{RespParser, RespValue};
use crate::storage::Database;

/// Lock a mutex, recovering the guard if the lock was poisoned by a panic in
/// another thread. AOF state (the buffered writer, pending bytes, and size
/// counter) is always left consistent after each critical section, so it is
/// safe to keep using it rather than propagating a poison panic onto the
/// command hot path.
#[inline]
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Fsync policy for AOF
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsyncPolicy {
    /// Fsync after every write
    Always,
    /// Fsync once per second (default)
    EverySecond,
    /// Let OS handle it
    No,
}

/// AOF (Append-Only File) persistence handler
pub struct AOF {
    path: String,
    file: Mutex<Option<BufWriter<File>>>,
    policy: FsyncPolicy,
    buffer: Mutex<Vec<u8>>,
    current_size: Mutex<usize>,
}

impl AOF {
    /// Create a new AOF handler
    pub fn new(path: String, policy: FsyncPolicy) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        let size = file.metadata()?.len() as usize;

        Ok(Self {
            path,
            file: Mutex::new(Some(BufWriter::new(file))),
            policy,
            buffer: Mutex::new(Vec::with_capacity(4096)),
            current_size: Mutex::new(size),
        })
    }

    /// Append a command to the AOF
    pub fn append(&self, command: &[RespValue]) -> io::Result<()> {
        if command.is_empty() {
            return Ok(());
        }

        // Serialize command to RESP
        let resp = self.command_to_resp(command);

        let mut buffer = lock(&self.buffer);
        buffer.extend_from_slice(&resp);

        match self.policy {
            FsyncPolicy::Always => {
                let mut file = lock(&self.file);
                if let Some(f) = file.as_mut() {
                    f.write_all(&buffer)?;
                    f.flush()?;
                    // Get inner file for sync
                    f.get_ref().sync_all()?;
                }
                buffer.clear();
            }
            FsyncPolicy::EverySecond | FsyncPolicy::No => {
                // Flush if buffer is large enough
                if buffer.len() > 4096 {
                    let mut file = lock(&self.file);
                    if let Some(f) = file.as_mut() {
                        f.write_all(&buffer)?;
                        f.flush()?;
                    }
                    buffer.clear();
                }
            }
        }

        let mut size = lock(&self.current_size);
        *size += resp.len();

        Ok(())
    }

    /// Flush buffer to disk
    pub fn flush(&self) -> io::Result<()> {
        let mut buffer = lock(&self.buffer);
        if buffer.is_empty() {
            return Ok(());
        }

        let mut file = lock(&self.file);
        if let Some(f) = file.as_mut() {
            f.write_all(&buffer)?;
            f.flush()?;
        }
        buffer.clear();
        Ok(())
    }

    /// Sync to disk
    pub fn sync(&self) -> io::Result<()> {
        self.flush()?;
        let file = lock(&self.file);
        if let Some(f) = file.as_ref() {
            f.get_ref().sync_all()?;
        }
        Ok(())
    }

    /// Load database from AOF file
    pub fn load(&self) -> io::Result<Database> {
        let path = Path::new(&self.path);
        if !path.exists() {
            return Ok(Database::new());
        }

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut db = Database::new();
        let mut parser = RespParser::new();

        // Read entire file
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;
        parser.feed(&data);

        // Parse and replay commands through the real command execution path so
        // that AOF replay has identical semantics to live command handling.
        loop {
            match parser.parse() {
                Ok(Some(value)) => {
                    if let Some(args) = value.into_array() {
                        if !args.is_empty() {
                            if let Some(cmd_name) = args[0].as_str() {
                                let cmd = cmd_name.to_uppercase();
                                let _ = CommandExecutor::execute(&cmd, &args[1..], &mut db);
                            }
                        }
                    }
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }

        Ok(db)
    }

    /// Rewrite AOF file to optimize size
    pub fn rewrite(&self, db: &Database) -> io::Result<()> {
        let temp_path = format!("{}.rewrite", self.path);
        let file = File::create(&temp_path)?;
        let mut writer = BufWriter::new(file);

        // Serialize the current dataset from an owned snapshot. This avoids the
        // previous const-to-mut pointer cast (undefined behavior) and needs no
        // deduplication since `snapshot` yields each live key exactly once.
        for (key, obj) in db.snapshot() {
            let commands = self.object_to_commands(&key, &obj);
            for cmd in commands {
                let resp = self.command_to_resp(&cmd);
                writer.write_all(&resp)?;
            }
        }

        writer.flush()?;
        drop(writer);

        // Close current file
        {
            let mut file = lock(&self.file);
            *file = None;
        }

        // Atomic rename
        std::fs::rename(&temp_path, &self.path)?;

        // Reopen file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        let size = file.metadata()?.len() as usize;

        {
            let mut f = lock(&self.file);
            *f = Some(BufWriter::new(file));
        }

        {
            let mut s = lock(&self.current_size);
            *s = size;
        }

        Ok(())
    }

    /// Convert command to RESP format
    fn command_to_resp(&self, args: &[RespValue]) -> Vec<u8> {
        let mut buf = bytes::BytesMut::with_capacity(64);
        RespValue::Array(Some(args.to_vec())).serialize_into(&mut buf);
        buf.to_vec()
    }

    /// Convert object to commands for AOF rewrite
    fn object_to_commands(&self, key: &str, obj: &crate::storage::RedisObject) -> Vec<Vec<RespValue>> {
        let mut commands = Vec::new();

        match obj {
            crate::storage::RedisObject::String(s) => {
                commands.push(vec![
                    RespValue::bulk_string("SET"),
                    RespValue::bulk_string(key),
                    RespValue::bulk(s.as_bytes()),
                ]);
            }
            crate::storage::RedisObject::List(list) => {
                if !list.is_empty() {
                    let mut cmd = vec![
                        RespValue::bulk_string("RPUSH"),
                        RespValue::bulk_string(key),
                    ];
                    for item in list {
                        cmd.push(RespValue::bulk(item.clone()));
                    }
                    commands.push(cmd);
                }
            }
            crate::storage::RedisObject::Set(set) => {
                if !set.is_empty() {
                    let mut cmd = vec![
                        RespValue::bulk_string("SADD"),
                        RespValue::bulk_string(key),
                    ];
                    for member in set {
                        cmd.push(RespValue::bulk(member.clone()));
                    }
                    commands.push(cmd);
                }
            }
            crate::storage::RedisObject::Hash(hash) => {
                if !hash.is_empty() {
                    let mut cmd = vec![
                        RespValue::bulk_string("HSET"),
                        RespValue::bulk_string(key),
                    ];
                    for (field, value) in hash {
                        cmd.push(RespValue::bulk(field.clone()));
                        cmd.push(RespValue::bulk(value.clone()));
                    }
                    commands.push(cmd);
                }
            }
            crate::storage::RedisObject::ZSet(zset) => {
                if !zset.is_empty() {
                    let mut cmd = vec![
                        RespValue::bulk_string("ZADD"),
                        RespValue::bulk_string(key),
                    ];
                    for (score, member) in &zset.sorted {
                        cmd.push(RespValue::bulk_string(score.to_string()));
                        cmd.push(RespValue::bulk(member.clone()));
                    }
                    commands.push(cmd);
                }
            }
        }

        commands
    }

    /// Get current AOF size
    pub fn size(&self) -> usize {
        *lock(&self.current_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_aof_append_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.aof");
        let aof = AOF::new(path.to_string_lossy().to_string(), FsyncPolicy::Always).unwrap();

        // Append SET command
        aof.append(&[
            RespValue::bulk_string("SET"),
            RespValue::bulk_string("key1"),
            RespValue::bulk_string("value1"),
        ]).unwrap();

        // Append another SET command
        aof.append(&[
            RespValue::bulk_string("SET"),
            RespValue::bulk_string("key2"),
            RespValue::bulk_string("value2"),
        ]).unwrap();

        aof.flush().unwrap();

        // Load and verify
        let mut db = aof.load().unwrap();

        assert_eq!(db.get_string("key1"), Some(b"value1".to_vec()));
        assert_eq!(db.get_string("key2"), Some(b"value2".to_vec()));
    }

    #[test]
    fn test_aof_fsync_policies() {
        let dir = tempdir().unwrap();

        // Test each policy
        for policy in [FsyncPolicy::Always, FsyncPolicy::EverySecond, FsyncPolicy::No] {
            let path = dir.path().join(format!("test_{:?}.aof", policy));
            let aof = AOF::new(path.to_string_lossy().to_string(), policy).unwrap();

            aof.append(&[
                RespValue::bulk_string("SET"),
                RespValue::bulk_string("key"),
                RespValue::bulk_string("value"),
            ]).unwrap();

            aof.flush().unwrap();
            aof.sync().unwrap();

            let db = aof.load().unwrap();
            assert_eq!(db.len(), 1);
        }
    }

    #[test]
    fn test_aof_empty_command() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.aof");
        let aof = AOF::new(path.to_string_lossy().to_string(), FsyncPolicy::Always).unwrap();

        // Append empty command should be no-op
        aof.append(&[]).unwrap();

        aof.flush().unwrap();

        let db = aof.load().unwrap();
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_aof_load_nonexistent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.aof");

        // Create AOF but don't write anything
        let aof = AOF::new(path.to_string_lossy().to_string(), FsyncPolicy::Always).unwrap();

        // Load should return empty database
        let db = aof.load().unwrap();
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_aof_list_commands() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.aof");
        let aof = AOF::new(path.to_string_lossy().to_string(), FsyncPolicy::Always).unwrap();

        // RPUSH command
        aof.append(&[
            RespValue::bulk_string("RPUSH"),
            RespValue::bulk_string("mylist"),
            RespValue::bulk_string("a"),
            RespValue::bulk_string("b"),
            RespValue::bulk_string("c"),
        ]).unwrap();

        aof.flush().unwrap();

        let mut db = aof.load().unwrap();

        if let Some(crate::storage::RedisObject::List(list)) = db.get("mylist") {
            assert_eq!(list.len(), 3);
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_aof_hash_commands() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.aof");
        let aof = AOF::new(path.to_string_lossy().to_string(), FsyncPolicy::Always).unwrap();

        // HSET command
        aof.append(&[
            RespValue::bulk_string("HSET"),
            RespValue::bulk_string("myhash"),
            RespValue::bulk_string("field1"),
            RespValue::bulk_string("value1"),
            RespValue::bulk_string("field2"),
            RespValue::bulk_string("value2"),
        ]).unwrap();

        aof.flush().unwrap();

        let mut db = aof.load().unwrap();

        if let Some(crate::storage::RedisObject::Hash(hash)) = db.get("myhash") {
            assert_eq!(hash.len(), 2);
        } else {
            panic!("Expected hash");
        }
    }

    #[test]
    fn test_aof_set_commands() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.aof");
        let aof = AOF::new(path.to_string_lossy().to_string(), FsyncPolicy::Always).unwrap();

        // SADD command
        aof.append(&[
            RespValue::bulk_string("SADD"),
            RespValue::bulk_string("myset"),
            RespValue::bulk_string("member1"),
            RespValue::bulk_string("member2"),
        ]).unwrap();

        aof.flush().unwrap();

        let mut db = aof.load().unwrap();

        if let Some(crate::storage::RedisObject::Set(set)) = db.get("myset") {
            assert_eq!(set.len(), 2);
        } else {
            panic!("Expected set");
        }
    }

    #[test]
    fn test_aof_zset_commands() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.aof");
        let aof = AOF::new(path.to_string_lossy().to_string(), FsyncPolicy::Always).unwrap();

        // ZADD command
        aof.append(&[
            RespValue::bulk_string("ZADD"),
            RespValue::bulk_string("myzset"),
            RespValue::bulk_string("1"),
            RespValue::bulk_string("one"),
            RespValue::bulk_string("2"),
            RespValue::bulk_string("two"),
        ]).unwrap();

        aof.flush().unwrap();

        let mut db = aof.load().unwrap();

        if let Some(crate::storage::RedisObject::ZSet(zset)) = db.get("myzset") {
            assert_eq!(zset.len(), 2);
        } else {
            panic!("Expected zset");
        }
    }

    #[test]
    fn test_aof_size_tracking() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.aof");
        let aof = AOF::new(path.to_string_lossy().to_string(), FsyncPolicy::Always).unwrap();

        let initial_size = aof.size();

        aof.append(&[
            RespValue::bulk_string("SET"),
            RespValue::bulk_string("key"),
            RespValue::bulk_string("value"),
        ]).unwrap();

        aof.flush().unwrap();

        assert!(aof.size() > initial_size);
    }

    #[test]
    fn test_aof_multiple_operations() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.aof");
        let aof = AOF::new(path.to_string_lossy().to_string(), FsyncPolicy::Always).unwrap();

        // Multiple SET operations
        for i in 0..10 {
            aof.append(&[
                RespValue::bulk_string("SET"),
                RespValue::bulk_string(format!("key{}", i)),
                RespValue::bulk_string(format!("value{}", i)),
            ]).unwrap();
        }

        aof.flush().unwrap();

        let mut db = aof.load().unwrap();

        // Verify all keys
        for i in 0..10 {
            assert_eq!(
                db.get_string(&format!("key{}", i)),
                Some(format!("value{}", i).into_bytes())
            );
        }
    }

    #[test]
    fn test_aof_overwrite() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.aof");
        let aof = AOF::new(path.to_string_lossy().to_string(), FsyncPolicy::Always).unwrap();

        // Set a key
        aof.append(&[
            RespValue::bulk_string("SET"),
            RespValue::bulk_string("key"),
            RespValue::bulk_string("value1"),
        ]).unwrap();

        // Overwrite the key
        aof.append(&[
            RespValue::bulk_string("SET"),
            RespValue::bulk_string("key"),
            RespValue::bulk_string("value2"),
        ]).unwrap();

        aof.flush().unwrap();

        let mut db = aof.load().unwrap();

        // Should have the latest value
        assert_eq!(db.get_string("key"), Some(b"value2".to_vec()));
    }

    #[test]
    fn test_aof_delete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.aof");
        let aof = AOF::new(path.to_string_lossy().to_string(), FsyncPolicy::Always).unwrap();

        // Set a key
        aof.append(&[
            RespValue::bulk_string("SET"),
            RespValue::bulk_string("key"),
            RespValue::bulk_string("value"),
        ]).unwrap();

        // Delete the key
        aof.append(&[
            RespValue::bulk_string("DEL"),
            RespValue::bulk_string("key"),
        ]).unwrap();

        aof.flush().unwrap();

        let mut db = aof.load().unwrap();

        // Key should not exist
        assert!(!db.exists("key"));
    }

    #[test]
    fn test_aof_incr_decr() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.aof");
        let aof = AOF::new(path.to_string_lossy().to_string(), FsyncPolicy::Always).unwrap();

        // INCR commands
        aof.append(&[
            RespValue::bulk_string("INCR"),
            RespValue::bulk_string("counter"),
        ]).unwrap();
        aof.append(&[
            RespValue::bulk_string("INCR"),
            RespValue::bulk_string("counter"),
        ]).unwrap();
        aof.append(&[
            RespValue::bulk_string("INCRBY"),
            RespValue::bulk_string("counter"),
            RespValue::bulk_string("10"),
        ]).unwrap();

        aof.flush().unwrap();

        let mut db = aof.load().unwrap();

        // Counter should be 12 (1 + 1 + 10)
        if let Some(crate::storage::RedisObject::String(s)) = db.get("counter") {
            assert_eq!(s.as_int(), Some(12));
        } else {
            panic!("Expected string");
        }
    }

    #[test]
    fn test_aof_buffer_flush_on_size() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.aof");
        let aof = AOF::new(path.to_string_lossy().to_string(), FsyncPolicy::EverySecond).unwrap();

        // Write enough data to trigger buffer flush
        let large_value = "x".repeat(5000);
        for i in 0..10 {
            aof.append(&[
                RespValue::bulk_string("SET"),
                RespValue::bulk_string(format!("key{}", i)),
                RespValue::bulk_string(&large_value),
            ]).unwrap();
        }

        aof.flush().unwrap();

        let db = aof.load().unwrap();
        assert_eq!(db.len(), 10);
    }
}
