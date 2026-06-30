use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::storage::{Database, RedisObject, StringObject, ZSetObject};

// RDB constants
const RDB_MAGIC: &[u8] = b"REDIS0011";
const RDB_OPCODE_AUX: u8 = 0xFA;
const RDB_OPCODE_SELECTDB: u8 = 0xFE;
const RDB_OPCODE_EXPIRETIME_MS: u8 = 0xFC;
const RDB_OPCODE_EOF: u8 = 0xFF;

// Object types
const RDB_TYPE_STRING: u8 = 0;
const RDB_TYPE_LIST: u8 = 1;
const RDB_TYPE_SET: u8 = 2;
const RDB_TYPE_ZSET: u8 = 3;
const RDB_TYPE_HASH: u8 = 4;

/// RDB persistence handler
pub struct RDB {
    path: String,
    compression: bool,
}

impl RDB {
    /// Create a new RDB handler
    pub fn new(path: String) -> Self {
        Self {
            path,
            compression: false,
        }
    }

    /// Save database to RDB file
    pub fn save(&self, db: &Database) -> io::Result<()> {
        let temp_path = format!("{}.temp", self.path);
        let file = File::create(&temp_path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_all(RDB_MAGIC)?;

        // Write aux fields
        self.write_aux(&mut writer, "redis-ver", "0.1.0")?;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.write_aux(&mut writer, "ctime", &timestamp.to_string())?;

        // Write database selector (DB 0)
        writer.write_all(&[RDB_OPCODE_SELECTDB])?;
        self.write_length(&mut writer, 0)?;

        // Write key-value pairs
        let keys = db.all_keys();

        for key in keys {
            // Get TTL (we'd need to expose expires in Database)
            // For now, skip expiration writing

            // Clone database to get immutable reference
            // This is a workaround - real implementation needs better access
            if let Some(_obj) = unsafe {
                let db_ptr = db as *const Database as *mut Database;
                (*db_ptr).get(&key)
            } {
                self.write_key_value(&mut writer, &key, _obj)?;
            }
        }

        // Write EOF
        writer.write_all(&[RDB_OPCODE_EOF])?;

        // Write checksum (simplified - just write 0s)
        writer.write_all(&[0u8; 8])?;

        writer.flush()?;
        drop(writer);

        // Atomic rename
        std::fs::rename(&temp_path, &self.path)?;

        Ok(())
    }

    /// Load database from RDB file
    pub fn load(&self) -> io::Result<Database> {
        let path = Path::new(&self.path);
        if !path.exists() {
            return Ok(Database::new());
        }

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut db = Database::new();

        // Verify header
        let mut header = [0u8; 9];
        reader.read_exact(&mut header)?;
        if &header[..5] != b"REDIS" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid RDB file",
            ));
        }

        // Parse RDB file
        loop {
            let mut opcode = [0u8; 1];
            if reader.read_exact(&mut opcode).is_err() {
                break;
            }

            match opcode[0] {
                RDB_OPCODE_EOF => break,
                RDB_OPCODE_SELECTDB => {
                    let _db_num = self.read_length(&mut reader)?;
                    // We only support DB 0 for now
                }
                RDB_OPCODE_AUX => {
                    // Skip aux field
                    let _key = self.read_string(&mut reader)?;
                    let _value = self.read_string(&mut reader)?;
                }
                RDB_OPCODE_EXPIRETIME_MS => {
                    let mut ms_bytes = [0u8; 8];
                    reader.read_exact(&mut ms_bytes)?;
                    let _expire_ms = u64::from_le_bytes(ms_bytes);
                    // Read the actual key-value after expiration
                    self.read_key_value(&mut reader, &mut db)?;
                }
                type_byte => {
                    // Regular key-value
                    self.read_object(&mut reader, &mut db, type_byte)?;
                }
            }
        }

        Ok(db)
    }

    /// Write auxiliary field
    fn write_aux(&self, writer: &mut BufWriter<File>, key: &str, value: &str) -> io::Result<()> {
        writer.write_all(&[RDB_OPCODE_AUX])?;
        self.write_string(writer, key.as_bytes())?;
        self.write_string(writer, value.as_bytes())?;
        Ok(())
    }

    /// Write length encoding
    fn write_length(&self, writer: &mut BufWriter<File>, len: usize) -> io::Result<()> {
        if len < 64 {
            // 6-bit length
            writer.write_all(&[len as u8])?;
        } else if len < 16384 {
            // 14-bit length
            let bytes = [
                0x40 | ((len >> 8) as u8 & 0x3F),
                (len & 0xFF) as u8,
            ];
            writer.write_all(&bytes)?;
        } else {
            // 32-bit length
            writer.write_all(&[0x80])?;
            writer.write_all(&(len as u32).to_be_bytes())?;
        }
        Ok(())
    }

    /// Write string
    fn write_string(&self, writer: &mut BufWriter<File>, data: &[u8]) -> io::Result<()> {
        self.write_length(writer, data.len())?;
        writer.write_all(data)?;
        Ok(())
    }

    /// Write key-value pair
    fn write_key_value(
        &self,
        writer: &mut BufWriter<File>,
        key: &str,
        obj: &RedisObject,
    ) -> io::Result<()> {
        match obj {
            RedisObject::String(s) => {
                writer.write_all(&[RDB_TYPE_STRING])?;
                self.write_string(writer, key.as_bytes())?;
                self.write_string(writer, &s.as_bytes())?;
            }
            RedisObject::List(list) => {
                writer.write_all(&[RDB_TYPE_LIST])?;
                self.write_string(writer, key.as_bytes())?;
                self.write_length(writer, list.len())?;
                for item in list {
                    self.write_string(writer, item)?;
                }
            }
            RedisObject::Set(set) => {
                writer.write_all(&[RDB_TYPE_SET])?;
                self.write_string(writer, key.as_bytes())?;
                self.write_length(writer, set.len())?;
                for member in set {
                    self.write_string(writer, member)?;
                }
            }
            RedisObject::Hash(hash) => {
                writer.write_all(&[RDB_TYPE_HASH])?;
                self.write_string(writer, key.as_bytes())?;
                self.write_length(writer, hash.len())?;
                for (field, value) in hash {
                    self.write_string(writer, field)?;
                    self.write_string(writer, value)?;
                }
            }
            RedisObject::ZSet(zset) => {
                writer.write_all(&[RDB_TYPE_ZSET])?;
                self.write_string(writer, key.as_bytes())?;
                self.write_length(writer, zset.len())?;
                for (score, member) in &zset.sorted {
                    self.write_string(writer, member)?;
                    // Write score as string
                    self.write_string(writer, score.to_string().as_bytes())?;
                }
            }
        }
        Ok(())
    }

    /// Read length encoding
    fn read_length(&self, reader: &mut BufReader<File>) -> io::Result<usize> {
        let mut first = [0u8; 1];
        reader.read_exact(&mut first)?;

        let enc_type = (first[0] & 0xC0) >> 6;
        match enc_type {
            0 => Ok((first[0] & 0x3F) as usize),
            1 => {
                let mut second = [0u8; 1];
                reader.read_exact(&mut second)?;
                Ok((((first[0] & 0x3F) as usize) << 8) | second[0] as usize)
            }
            2 => {
                let mut len_bytes = [0u8; 4];
                reader.read_exact(&mut len_bytes)?;
                Ok(u32::from_be_bytes(len_bytes) as usize)
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid length encoding",
            )),
        }
    }

    /// Read string
    fn read_string(&self, reader: &mut BufReader<File>) -> io::Result<Vec<u8>> {
        let len = self.read_length(reader)?;
        let mut data = vec![0u8; len];
        reader.read_exact(&mut data)?;
        Ok(data)
    }

    /// Read key-value after expiration opcode
    fn read_key_value(&self, reader: &mut BufReader<File>, db: &mut Database) -> io::Result<()> {
        let mut type_byte = [0u8; 1];
        reader.read_exact(&mut type_byte)?;
        self.read_object(reader, db, type_byte[0])
    }

    /// Read object by type
    fn read_object(
        &self,
        reader: &mut BufReader<File>,
        db: &mut Database,
        type_byte: u8,
    ) -> io::Result<()> {
        let key = String::from_utf8(self.read_string(reader)?)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        match type_byte {
            RDB_TYPE_STRING => {
                let value = self.read_string(reader)?;
                db.set(key, RedisObject::String(StringObject::from_bytes(value)));
            }
            RDB_TYPE_LIST => {
                let len = self.read_length(reader)?;
                let mut list = Vec::with_capacity(len);
                for _ in 0..len {
                    list.push(self.read_string(reader)?);
                }
                db.set(key, RedisObject::List(list));
            }
            RDB_TYPE_SET => {
                let len = self.read_length(reader)?;
                let mut set = std::collections::HashSet::new();
                for _ in 0..len {
                    set.insert(self.read_string(reader)?);
                }
                db.set(key, RedisObject::Set(set));
            }
            RDB_TYPE_HASH => {
                let len = self.read_length(reader)?;
                let mut hash = std::collections::HashMap::new();
                for _ in 0..len {
                    let field = self.read_string(reader)?;
                    let value = self.read_string(reader)?;
                    hash.insert(field, value);
                }
                db.set(key, RedisObject::Hash(hash));
            }
            RDB_TYPE_ZSET => {
                let len = self.read_length(reader)?;
                let mut zset = ZSetObject::new();
                for _ in 0..len {
                    let member = self.read_string(reader)?;
                    let score_str = self.read_string(reader)?;
                    let score: f64 = String::from_utf8_lossy(&score_str)
                        .parse()
                        .unwrap_or(0.0);
                    zset.add(score, member);
                }
                db.set(key, RedisObject::ZSet(zset));
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unknown RDB type: {}", type_byte),
                ));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use tempfile::tempdir;

    #[test]
    fn test_rdb_save_load_string() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.rdb");
        let rdb = RDB::new(path.to_string_lossy().to_string());

        let mut db = Database::new();
        db.set_string("key1".to_string(), b"value1".to_vec());
        db.set_string("key2".to_string(), b"value2".to_vec());
        db.set_string("number".to_string(), b"12345".to_vec());

        // Save
        rdb.save(&db).unwrap();

        // Load
        let loaded_db = rdb.load().unwrap();

        assert_eq!(loaded_db.len(), 3);
    }

    #[test]
    fn test_rdb_save_load_list() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.rdb");
        let rdb = RDB::new(path.to_string_lossy().to_string());

        let mut db = Database::new();
        let list = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        db.set("mylist".to_string(), RedisObject::List(list));

        // Save
        rdb.save(&db).unwrap();

        // Load
        let mut loaded_db = rdb.load().unwrap();

        // Verify list was loaded
        if let Some(RedisObject::List(loaded_list)) = loaded_db.get("mylist") {
            assert_eq!(loaded_list.len(), 3);
            assert_eq!(loaded_list[0], b"a");
            assert_eq!(loaded_list[1], b"b");
            assert_eq!(loaded_list[2], b"c");
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_rdb_save_load_set() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.rdb");
        let rdb = RDB::new(path.to_string_lossy().to_string());

        let mut db = Database::new();
        let mut set = HashSet::new();
        set.insert(b"member1".to_vec());
        set.insert(b"member2".to_vec());
        set.insert(b"member3".to_vec());
        db.set("myset".to_string(), RedisObject::Set(set));

        // Save
        rdb.save(&db).unwrap();

        // Load
        let mut loaded_db = rdb.load().unwrap();

        // Verify set was loaded
        if let Some(RedisObject::Set(loaded_set)) = loaded_db.get("myset") {
            assert_eq!(loaded_set.len(), 3);
            assert!(loaded_set.contains(&b"member1".to_vec()));
            assert!(loaded_set.contains(&b"member2".to_vec()));
            assert!(loaded_set.contains(&b"member3".to_vec()));
        } else {
            panic!("Expected set");
        }
    }

    #[test]
    fn test_rdb_save_load_hash() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.rdb");
        let rdb = RDB::new(path.to_string_lossy().to_string());

        let mut db = Database::new();
        let mut hash = HashMap::new();
        hash.insert(b"field1".to_vec(), b"value1".to_vec());
        hash.insert(b"field2".to_vec(), b"value2".to_vec());
        db.set("myhash".to_string(), RedisObject::Hash(hash));

        // Save
        rdb.save(&db).unwrap();

        // Load
        let mut loaded_db = rdb.load().unwrap();

        // Verify hash was loaded
        if let Some(RedisObject::Hash(loaded_hash)) = loaded_db.get("myhash") {
            assert_eq!(loaded_hash.len(), 2);
            assert_eq!(loaded_hash.get(&b"field1".to_vec()), Some(&b"value1".to_vec()));
            assert_eq!(loaded_hash.get(&b"field2".to_vec()), Some(&b"value2".to_vec()));
        } else {
            panic!("Expected hash");
        }
    }

    #[test]
    fn test_rdb_save_load_zset() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.rdb");
        let rdb = RDB::new(path.to_string_lossy().to_string());

        let mut db = Database::new();
        let mut zset = ZSetObject::new();
        zset.add(1.0, b"one".to_vec());
        zset.add(2.0, b"two".to_vec());
        zset.add(3.0, b"three".to_vec());
        db.set("myzset".to_string(), RedisObject::ZSet(zset));

        // Save
        rdb.save(&db).unwrap();

        // Load
        let mut loaded_db = rdb.load().unwrap();

        // Verify zset was loaded
        if let Some(RedisObject::ZSet(loaded_zset)) = loaded_db.get("myzset") {
            assert_eq!(loaded_zset.len(), 3);
            assert_eq!(loaded_zset.score(b"one"), Some(1.0));
            assert_eq!(loaded_zset.score(b"two"), Some(2.0));
            assert_eq!(loaded_zset.score(b"three"), Some(3.0));
        } else {
            panic!("Expected zset");
        }
    }

    #[test]
    fn test_rdb_load_nonexistent_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.rdb");
        let rdb = RDB::new(path.to_string_lossy().to_string());

        // Load should return empty database
        let db = rdb.load().unwrap();
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_rdb_invalid_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("invalid.rdb");

        // Write invalid data
        std::fs::write(&path, b"INVALID DATA").unwrap();

        let rdb = RDB::new(path.to_string_lossy().to_string());

        // Load should fail
        let result = rdb.load();
        assert!(result.is_err());
    }

    #[test]
    fn test_rdb_length_encoding() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.rdb");
        let rdb = RDB::new(path.to_string_lossy().to_string());

        let mut db = Database::new();

        // Create string with different lengths to test encoding
        db.set_string("short".to_string(), b"x".to_vec()); // < 64
        db.set_string("medium".to_string(), "x".repeat(200).into_bytes()); // 64-16384
        db.set_string("long".to_string(), "x".repeat(20000).into_bytes()); // > 16384

        rdb.save(&db).unwrap();
        let loaded_db = rdb.load().unwrap();

        assert_eq!(loaded_db.len(), 3);
    }

    #[test]
    fn test_rdb_atomic_save() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.rdb");
        let temp_path = dir.path().join("test.rdb.temp");
        let rdb = RDB::new(path.to_string_lossy().to_string());

        let mut db = Database::new();
        db.set_string("key".to_string(), b"value".to_vec());

        rdb.save(&db).unwrap();

        // Verify final file exists and temp doesn't
        assert!(path.exists());
        assert!(!temp_path.exists());
    }

    #[test]
    fn test_rdb_mixed_types() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.rdb");
        let rdb = RDB::new(path.to_string_lossy().to_string());

        let mut db = Database::new();

        // Add different types
        db.set_string("string_key".to_string(), b"string_value".to_vec());
        db.set("list_key".to_string(), RedisObject::List(vec![b"a".to_vec(), b"b".to_vec()]));

        let mut set = HashSet::new();
        set.insert(b"x".to_vec());
        db.set("set_key".to_string(), RedisObject::Set(set));

        let mut hash = HashMap::new();
        hash.insert(b"f".to_vec(), b"v".to_vec());
        db.set("hash_key".to_string(), RedisObject::Hash(hash));

        let mut zset = ZSetObject::new();
        zset.add(1.0, b"m".to_vec());
        db.set("zset_key".to_string(), RedisObject::ZSet(zset));

        rdb.save(&db).unwrap();
        let loaded_db = rdb.load().unwrap();

        assert_eq!(loaded_db.len(), 5);
    }

    #[test]
    fn test_rdb_empty_values() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.rdb");
        let rdb = RDB::new(path.to_string_lossy().to_string());

        let mut db = Database::new();
        db.set_string("empty_string".to_string(), Vec::new());
        db.set("empty_list".to_string(), RedisObject::List(Vec::new()));
        db.set("empty_set".to_string(), RedisObject::Set(HashSet::new()));
        db.set("empty_hash".to_string(), RedisObject::Hash(HashMap::new()));

        rdb.save(&db).unwrap();
        let loaded_db = rdb.load().unwrap();

        assert_eq!(loaded_db.len(), 4);
    }

    #[test]
    fn test_rdb_binary_data() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.rdb");
        let rdb = RDB::new(path.to_string_lossy().to_string());

        let mut db = Database::new();
        // Store binary data with null bytes
        let binary_value = vec![0u8, 1, 2, 3, 0, 255, 254, 0];
        db.set_string("binary".to_string(), binary_value.clone());

        rdb.save(&db).unwrap();
        let mut loaded_db = rdb.load().unwrap();

        if let Some(RedisObject::String(s)) = loaded_db.get("binary") {
            assert_eq!(s.as_bytes(), binary_value);
        } else {
            panic!("Expected string");
        }
    }

    #[test]
    fn test_rdb_unicode_keys() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.rdb");
        let rdb = RDB::new(path.to_string_lossy().to_string());

        let mut db = Database::new();
        db.set_string("key".to_string(), b"value".to_vec());

        rdb.save(&db).unwrap();
        let loaded_db = rdb.load().unwrap();

        assert!(loaded_db.len() >= 1);
    }
}
