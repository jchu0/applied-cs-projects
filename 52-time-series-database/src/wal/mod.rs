//! Write-Ahead Log (WAL) for durability
//!
//! The WAL ensures durability by writing all operations to disk
//! before they are applied to the in-memory structures.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher;
use parking_lot::Mutex;

use crate::error::{Result, TsdbError};
use crate::types::{DataPoint, Metric, Tags};
use crate::compression::varint::{encode_varint, decode_varint};

/// WAL entry type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalEntryType {
    /// Write a single data point
    Write = 1,
    /// Write a batch of data points
    WriteBatch = 2,
    /// Checkpoint (for recovery)
    Checkpoint = 3,
}

impl TryFrom<u8> for WalEntryType {
    type Error = TsdbError;

    fn try_from(value: u8) -> std::result::Result<Self, Self::Error> {
        match value {
            1 => Ok(WalEntryType::Write),
            2 => Ok(WalEntryType::WriteBatch),
            3 => Ok(WalEntryType::Checkpoint),
            _ => Err(TsdbError::wal(format!("Unknown WAL entry type: {}", value))),
        }
    }
}

/// A single WAL entry
#[derive(Debug, Clone)]
pub enum WalEntry {
    /// Single write operation
    Write {
        metric_name: String,
        tags: Tags,
        timestamp: i64,
        value: f64,
    },
    /// Batch write operation
    WriteBatch {
        points: Vec<(String, Tags, i64, f64)>,
    },
    /// Checkpoint marker
    Checkpoint {
        sequence: u64,
    },
}

impl WalEntry {
    /// Serialize the entry to bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        match self {
            WalEntry::Write {
                metric_name,
                tags,
                timestamp,
                value,
            } => {
                buf.push(WalEntryType::Write as u8);

                // Metric name
                let name_bytes = metric_name.as_bytes();
                buf.extend_from_slice(&encode_varint(name_bytes.len() as u64));
                buf.extend_from_slice(name_bytes);

                // Tags
                buf.extend_from_slice(&encode_varint(tags.len() as u64));
                for (k, v) in tags {
                    let k_bytes = k.as_bytes();
                    let v_bytes = v.as_bytes();
                    buf.extend_from_slice(&encode_varint(k_bytes.len() as u64));
                    buf.extend_from_slice(k_bytes);
                    buf.extend_from_slice(&encode_varint(v_bytes.len() as u64));
                    buf.extend_from_slice(v_bytes);
                }

                // Timestamp and value
                buf.extend_from_slice(&timestamp.to_le_bytes());
                buf.extend_from_slice(&value.to_le_bytes());
            }
            WalEntry::WriteBatch { points } => {
                buf.push(WalEntryType::WriteBatch as u8);
                buf.extend_from_slice(&encode_varint(points.len() as u64));

                for (name, tags, ts, val) in points {
                    let name_bytes = name.as_bytes();
                    buf.extend_from_slice(&encode_varint(name_bytes.len() as u64));
                    buf.extend_from_slice(name_bytes);

                    buf.extend_from_slice(&encode_varint(tags.len() as u64));
                    for (k, v) in tags {
                        let k_bytes = k.as_bytes();
                        let v_bytes = v.as_bytes();
                        buf.extend_from_slice(&encode_varint(k_bytes.len() as u64));
                        buf.extend_from_slice(k_bytes);
                        buf.extend_from_slice(&encode_varint(v_bytes.len() as u64));
                        buf.extend_from_slice(v_bytes);
                    }

                    buf.extend_from_slice(&ts.to_le_bytes());
                    buf.extend_from_slice(&val.to_le_bytes());
                }
            }
            WalEntry::Checkpoint { sequence } => {
                buf.push(WalEntryType::Checkpoint as u8);
                buf.extend_from_slice(&sequence.to_le_bytes());
            }
        }

        buf
    }

    /// Deserialize an entry from bytes
    pub fn deserialize(data: &[u8]) -> Result<(Self, usize)> {
        if data.is_empty() {
            return Err(TsdbError::wal("Empty WAL entry"));
        }

        let entry_type = WalEntryType::try_from(data[0])?;
        let mut offset = 1;

        let entry = match entry_type {
            WalEntryType::Write => {
                // Metric name
                let (name_len, bytes_read) = decode_varint(&data[offset..])?;
                offset += bytes_read;
                let metric_name = String::from_utf8(data[offset..offset + name_len as usize].to_vec())
                    .map_err(|e| TsdbError::wal(format!("Invalid UTF-8: {}", e)))?;
                offset += name_len as usize;

                // Tags
                let (tags_count, bytes_read) = decode_varint(&data[offset..])?;
                offset += bytes_read;
                let mut tags = Tags::new();
                for _ in 0..tags_count {
                    let (k_len, bytes_read) = decode_varint(&data[offset..])?;
                    offset += bytes_read;
                    let key = String::from_utf8(data[offset..offset + k_len as usize].to_vec())
                        .map_err(|e| TsdbError::wal(format!("Invalid UTF-8: {}", e)))?;
                    offset += k_len as usize;

                    let (v_len, bytes_read) = decode_varint(&data[offset..])?;
                    offset += bytes_read;
                    let value = String::from_utf8(data[offset..offset + v_len as usize].to_vec())
                        .map_err(|e| TsdbError::wal(format!("Invalid UTF-8: {}", e)))?;
                    offset += v_len as usize;

                    tags.insert(key, value);
                }

                // Timestamp and value
                let timestamp = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                offset += 8;
                let value = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                offset += 8;

                WalEntry::Write {
                    metric_name,
                    tags,
                    timestamp,
                    value,
                }
            }
            WalEntryType::WriteBatch => {
                let (count, bytes_read) = decode_varint(&data[offset..])?;
                offset += bytes_read;

                let mut points = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    let (name_len, bytes_read) = decode_varint(&data[offset..])?;
                    offset += bytes_read;
                    let name = String::from_utf8(data[offset..offset + name_len as usize].to_vec())
                        .map_err(|e| TsdbError::wal(format!("Invalid UTF-8: {}", e)))?;
                    offset += name_len as usize;

                    let (tags_count, bytes_read) = decode_varint(&data[offset..])?;
                    offset += bytes_read;
                    let mut tags = Tags::new();
                    for _ in 0..tags_count {
                        let (k_len, bytes_read) = decode_varint(&data[offset..])?;
                        offset += bytes_read;
                        let key = String::from_utf8(data[offset..offset + k_len as usize].to_vec())
                            .map_err(|e| TsdbError::wal(format!("Invalid UTF-8: {}", e)))?;
                        offset += k_len as usize;

                        let (v_len, bytes_read) = decode_varint(&data[offset..])?;
                        offset += bytes_read;
                        let value = String::from_utf8(data[offset..offset + v_len as usize].to_vec())
                            .map_err(|e| TsdbError::wal(format!("Invalid UTF-8: {}", e)))?;
                        offset += v_len as usize;

                        tags.insert(key, value);
                    }

                    let timestamp = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                    offset += 8;
                    let value = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                    offset += 8;

                    points.push((name, tags, timestamp, value));
                }

                WalEntry::WriteBatch { points }
            }
            WalEntryType::Checkpoint => {
                let sequence = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                offset += 8;
                WalEntry::Checkpoint { sequence }
            }
        };

        Ok((entry, offset))
    }
}

/// WAL segment file
const WAL_MAGIC: u32 = 0x57414C31; // "WAL1"

/// Write-Ahead Log
#[derive(Debug)]
pub struct WriteAheadLog {
    /// Directory for WAL files
    dir: PathBuf,
    /// Current segment writer
    writer: Mutex<Option<BufWriter<File>>>,
    /// Current segment number
    segment_number: AtomicU64,
    /// Current sequence number
    sequence: AtomicU64,
    /// Maximum segment size
    max_segment_size: u64,
    /// Current segment size
    current_size: AtomicU64,
    /// Sync on every write
    sync_on_write: bool,
}

impl WriteAheadLog {
    /// Create a new WAL
    pub fn new<P: AsRef<Path>>(dir: P) -> Result<Self> {
        Self::with_options(dir, 64 * 1024 * 1024, false) // 64MB segments
    }

    /// Create a WAL with custom options
    pub fn with_options<P: AsRef<Path>>(
        dir: P,
        max_segment_size: u64,
        sync_on_write: bool,
    ) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;

        let wal = Self {
            dir,
            writer: Mutex::new(None),
            segment_number: AtomicU64::new(0),
            sequence: AtomicU64::new(0),
            max_segment_size,
            current_size: AtomicU64::new(0),
            sync_on_write,
        };

        wal.recover_state()?;
        wal.ensure_segment()?;

        Ok(wal)
    }

    /// Recover state from existing WAL files
    fn recover_state(&self) -> Result<()> {
        let mut max_segment = 0u64;
        let mut max_sequence = 0u64;

        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("wal_") && name.ends_with(".log") {
                    if let Ok(num) = name[4..name.len() - 4].parse::<u64>() {
                        max_segment = max_segment.max(num);

                        // Scan for max sequence
                        if let Ok(entries) = self.read_segment(&path) {
                            for entry in entries {
                                if let WalEntry::Checkpoint { sequence } = entry {
                                    max_sequence = max_sequence.max(sequence);
                                }
                            }
                        }
                    }
                }
            }
        }

        self.segment_number.store(max_segment + 1, Ordering::SeqCst);
        self.sequence.store(max_sequence, Ordering::SeqCst);

        Ok(())
    }

    /// Ensure a segment is open for writing
    fn ensure_segment(&self) -> Result<()> {
        let mut writer = self.writer.lock();

        if writer.is_none() {
            let segment_num = self.segment_number.load(Ordering::SeqCst);
            let path = self.segment_path(segment_num);

            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .append(true)
                .open(&path)?;

            let mut buf_writer = BufWriter::new(file);

            // Write magic header for new file
            if std::fs::metadata(&path)?.len() == 0 {
                buf_writer.write_u32::<LittleEndian>(WAL_MAGIC)?;
                buf_writer.flush()?;
                self.current_size.store(4, Ordering::SeqCst);
            }

            *writer = Some(buf_writer);
        }

        Ok(())
    }

    /// Get the path for a segment
    fn segment_path(&self, segment_num: u64) -> PathBuf {
        self.dir.join(format!("wal_{:010}.log", segment_num))
    }

    /// Write an entry to the WAL
    pub fn append(&self, entry: &WalEntry) -> Result<u64> {
        self.ensure_segment()?;

        let data = entry.serialize();
        let mut frame = Vec::new();

        // Frame format: length (4 bytes) + data + checksum (4 bytes)
        frame.extend_from_slice(&(data.len() as u32).to_le_bytes());
        frame.extend_from_slice(&data);

        let mut hasher = Hasher::new();
        hasher.update(&data);
        let checksum = hasher.finalize();
        frame.extend_from_slice(&checksum.to_le_bytes());

        {
            let mut writer = self.writer.lock();
            if let Some(ref mut w) = *writer {
                w.write_all(&frame)?;

                if self.sync_on_write {
                    w.flush()?;
                    w.get_ref().sync_data()?;
                }
            }
        }

        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst);
        let new_size = self.current_size.fetch_add(frame.len() as u64, Ordering::SeqCst);

        // Check if we need to rotate (lock released before rotate to avoid deadlock)
        if new_size + frame.len() as u64 > self.max_segment_size {
            self.rotate()?;
        }

        Ok(sequence)
    }

    /// Rotate to a new segment
    fn rotate(&self) -> Result<()> {
        let mut writer = self.writer.lock();

        if let Some(ref mut w) = *writer {
            w.flush()?;
        }

        *writer = None;
        self.segment_number.fetch_add(1, Ordering::SeqCst);
        self.current_size.store(0, Ordering::SeqCst);

        drop(writer);
        self.ensure_segment()?;

        Ok(())
    }

    /// Sync the WAL to disk
    pub fn sync(&self) -> Result<()> {
        let mut writer = self.writer.lock();
        if let Some(ref mut w) = *writer {
            w.flush()?;
            w.get_ref().sync_data()?;
        }
        Ok(())
    }

    /// Read entries from a segment file
    fn read_segment(&self, path: &Path) -> Result<Vec<WalEntry>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Check magic
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != WAL_MAGIC {
            return Err(TsdbError::wal("Invalid WAL magic"));
        }

        let mut entries = Vec::new();

        loop {
            // Read frame length
            let length = match reader.read_u32::<LittleEndian>() {
                Ok(l) => l,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            };

            // Read data
            let mut data = vec![0u8; length as usize];
            reader.read_exact(&mut data)?;

            // Read and verify checksum
            let stored_checksum = reader.read_u32::<LittleEndian>()?;
            let mut hasher = Hasher::new();
            hasher.update(&data);
            let actual_checksum = hasher.finalize();

            if stored_checksum != actual_checksum {
                return Err(TsdbError::ChecksumMismatch {
                    expected: stored_checksum,
                    actual: actual_checksum,
                });
            }

            let (entry, _) = WalEntry::deserialize(&data)?;
            entries.push(entry);
        }

        Ok(entries)
    }

    /// Replay all WAL entries
    pub fn replay<F>(&self, mut callback: F) -> Result<u64>
    where
        F: FnMut(WalEntry) -> Result<()>,
    {
        let mut count = 0;

        // Get sorted list of segment files
        let mut segments: Vec<PathBuf> = std::fs::read_dir(&self.dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .map_or(false, |ext| ext == "log")
            })
            .collect();
        segments.sort();

        for path in segments {
            let entries = self.read_segment(&path)?;
            for entry in entries {
                callback(entry)?;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Truncate WAL after a checkpoint
    pub fn truncate_before(&self, segment_num: u64) -> Result<usize> {
        let mut deleted = 0;

        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("wal_") && name.ends_with(".log") {
                    if let Ok(num) = name[4..name.len() - 4].parse::<u64>() {
                        if num < segment_num {
                            std::fs::remove_file(&path)?;
                            deleted += 1;
                        }
                    }
                }
            }
        }

        Ok(deleted)
    }

    /// Get current sequence number
    pub fn sequence(&self) -> u64 {
        self.sequence.load(Ordering::SeqCst)
    }

    /// Get current segment number
    pub fn segment(&self) -> u64 {
        self.segment_number.load(Ordering::SeqCst)
    }

    /// Close the WAL
    pub fn close(&self) -> Result<()> {
        self.sync()?;
        let mut writer = self.writer.lock();
        *writer = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_wal_entry_serialize_write() {
        let mut tags = Tags::new();
        tags.insert("host".into(), "server1".into());

        let entry = WalEntry::Write {
            metric_name: "cpu.usage".into(),
            tags,
            timestamp: 1000,
            value: 42.5,
        };

        let serialized = entry.serialize();
        let (deserialized, _) = WalEntry::deserialize(&serialized).unwrap();

        if let WalEntry::Write {
            metric_name,
            tags: dtags,
            timestamp,
            value,
        } = deserialized
        {
            assert_eq!(metric_name, "cpu.usage");
            assert_eq!(dtags.get("host"), Some(&"server1".to_string()));
            assert_eq!(timestamp, 1000);
            assert_eq!(value, 42.5);
        } else {
            panic!("Wrong entry type");
        }
    }

    #[test]
    fn test_wal_entry_serialize_batch() {
        let mut tags = Tags::new();
        tags.insert("env".into(), "prod".into());

        let entry = WalEntry::WriteBatch {
            points: vec![
                ("metric1".into(), tags.clone(), 100, 1.0),
                ("metric2".into(), tags.clone(), 200, 2.0),
            ],
        };

        let serialized = entry.serialize();
        let (deserialized, _) = WalEntry::deserialize(&serialized).unwrap();

        if let WalEntry::WriteBatch { points } = deserialized {
            assert_eq!(points.len(), 2);
            assert_eq!(points[0].0, "metric1");
            assert_eq!(points[1].0, "metric2");
        } else {
            panic!("Wrong entry type");
        }
    }

    #[test]
    fn test_wal_entry_serialize_checkpoint() {
        let entry = WalEntry::Checkpoint { sequence: 12345 };

        let serialized = entry.serialize();
        let (deserialized, _) = WalEntry::deserialize(&serialized).unwrap();

        if let WalEntry::Checkpoint { sequence } = deserialized {
            assert_eq!(sequence, 12345);
        } else {
            panic!("Wrong entry type");
        }
    }

    #[test]
    fn test_wal_append_and_replay() {
        let dir = tempdir().unwrap();
        let wal = WriteAheadLog::new(dir.path()).unwrap();

        let tags = Tags::new();

        // Append some entries
        wal.append(&WalEntry::Write {
            metric_name: "test".into(),
            tags: tags.clone(),
            timestamp: 100,
            value: 1.0,
        })
        .unwrap();

        wal.append(&WalEntry::Write {
            metric_name: "test".into(),
            tags: tags.clone(),
            timestamp: 200,
            value: 2.0,
        })
        .unwrap();

        wal.sync().unwrap();

        // Replay
        let mut count = 0;
        wal.replay(|entry| {
            if let WalEntry::Write { .. } = entry {
                count += 1;
            }
            Ok(())
        })
        .unwrap();

        assert_eq!(count, 2);
    }

    #[test]
    fn test_wal_recovery() {
        let dir = tempdir().unwrap();

        // Write some entries
        {
            let wal = WriteAheadLog::new(dir.path()).unwrap();
            let tags = Tags::new();

            for i in 0..10 {
                wal.append(&WalEntry::Write {
                    metric_name: "test".into(),
                    tags: tags.clone(),
                    timestamp: i * 100,
                    value: i as f64,
                })
                .unwrap();
            }

            wal.sync().unwrap();
        }

        // Recover and verify
        {
            let wal = WriteAheadLog::new(dir.path()).unwrap();
            let mut count = 0;
            wal.replay(|_| {
                count += 1;
                Ok(())
            })
            .unwrap();

            assert_eq!(count, 10);
        }
    }

    #[test]
    fn test_wal_sync_on_write() {
        let dir = tempdir().unwrap();
        let wal = WriteAheadLog::with_options(dir.path(), 1024 * 1024, true).unwrap();

        let tags = Tags::new();
        wal.append(&WalEntry::Write {
            metric_name: "test".into(),
            tags,
            timestamp: 100,
            value: 1.0,
        })
        .unwrap();

        // File should be synced immediately
        let mut count = 0;
        wal.replay(|_| {
            count += 1;
            Ok(())
        })
        .unwrap();

        assert_eq!(count, 1);
    }

    #[test]
    fn test_wal_truncate() {
        let dir = tempdir().unwrap();
        let wal = WriteAheadLog::with_options(dir.path(), 100, false).unwrap(); // Small segments

        let tags = Tags::new();

        // Write enough to create multiple segments
        for i in 0..50 {
            wal.append(&WalEntry::Write {
                metric_name: format!("test{}", i),
                tags: tags.clone(),
                timestamp: i * 100,
                value: i as f64,
            })
            .unwrap();
        }

        wal.sync().unwrap();

        // Should have multiple segments
        let current_segment = wal.segment();
        assert!(current_segment > 0);

        // Truncate old segments
        let deleted = wal.truncate_before(current_segment).unwrap();
        assert!(deleted > 0);
    }

    #[test]
    fn test_wal_close() {
        let dir = tempdir().unwrap();
        let wal = WriteAheadLog::new(dir.path()).unwrap();

        let tags = Tags::new();
        wal.append(&WalEntry::Write {
            metric_name: "test".into(),
            tags,
            timestamp: 100,
            value: 1.0,
        })
        .unwrap();

        wal.close().unwrap();

        // Can still create new WAL and read
        let wal2 = WriteAheadLog::new(dir.path()).unwrap();
        let mut count = 0;
        wal2.replay(|_| {
            count += 1;
            Ok(())
        })
        .unwrap();

        assert_eq!(count, 1);
    }
}
