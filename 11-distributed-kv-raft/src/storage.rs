//! Storage components: Key-Value FSM and Write-Ahead Log.

use crate::error::{Error, Result};
use crate::node::{ApplyResult, Command, LogEntry};
use crate::LogIndex;

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write as IoWrite};
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

/// Trait for persistent storage.
pub trait Storage: Send + Sync {
    /// Append entries to the log.
    fn append(&mut self, entries: &[LogEntry]) -> Result<()>;

    /// Get entries from the log.
    fn get_entries(&self, from: LogIndex, to: LogIndex) -> Result<Vec<LogEntry>>;

    /// Truncate log from given index.
    fn truncate(&mut self, from: LogIndex) -> Result<()>;

    /// Get last entry.
    fn last_entry(&self) -> Option<&LogEntry>;

    /// Compact log up to given index (for snapshots).
    fn compact(&mut self, up_to: LogIndex) -> Result<()>;
}

/// In-memory storage for testing.
pub struct MemoryStorage {
    entries: Vec<LogEntry>,
}

impl MemoryStorage {
    /// Create new memory storage.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl Storage for MemoryStorage {
    fn append(&mut self, entries: &[LogEntry]) -> Result<()> {
        self.entries.extend(entries.iter().cloned());
        Ok(())
    }

    fn get_entries(&self, from: LogIndex, to: LogIndex) -> Result<Vec<LogEntry>> {
        Ok(self
            .entries
            .iter()
            .filter(|e| e.index >= from && e.index <= to)
            .cloned()
            .collect())
    }

    fn truncate(&mut self, from: LogIndex) -> Result<()> {
        self.entries.retain(|e| e.index < from);
        Ok(())
    }

    fn last_entry(&self) -> Option<&LogEntry> {
        self.entries.last()
    }

    fn compact(&mut self, up_to: LogIndex) -> Result<()> {
        self.entries.retain(|e| e.index > up_to);
        Ok(())
    }
}

/// Key-Value Finite State Machine.
pub struct KeyValueFSM {
    /// The actual key-value data.
    data: HashMap<Vec<u8>, Vec<u8>>,
    /// Last applied log index.
    last_applied_index: LogIndex,
    /// Last applied log term.
    last_applied_term: u64,
}

impl KeyValueFSM {
    /// Create a new Key-Value FSM.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            last_applied_index: 0,
            last_applied_term: 0,
        }
    }

    /// Apply a log entry to the state machine.
    pub fn apply(&mut self, entry: &LogEntry) -> ApplyResult {
        self.last_applied_index = entry.index;
        self.last_applied_term = entry.term;

        match &entry.command {
            Command::Put { key, value } => {
                self.data.insert(key.clone(), value.clone());
                ApplyResult::Success
            }
            Command::Delete { key } => {
                self.data.remove(key);
                ApplyResult::Success
            }
            Command::Get { key } => {
                let value = self.data.get(key).cloned();
                ApplyResult::Value(value)
            }
            Command::NoOp => ApplyResult::Success,
        }
    }

    /// Get a value from the state machine.
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.data.get(key).cloned()
    }

    /// Create a snapshot of the current state.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            last_included_index: self.last_applied_index,
            last_included_term: self.last_applied_term,
            data: bincode::serialize(&self.data).unwrap_or_default(),
        }
    }

    /// Restore state from a snapshot.
    pub fn restore(&mut self, snapshot: &Snapshot) -> Result<()> {
        self.data = bincode::deserialize(&snapshot.data)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        self.last_applied_index = snapshot.last_included_index;
        self.last_applied_term = snapshot.last_included_term;
        Ok(())
    }

    /// Get last applied index.
    pub fn last_applied_index(&self) -> LogIndex {
        self.last_applied_index
    }

    /// Get last applied term.
    pub fn last_applied_term(&self) -> u64 {
        self.last_applied_term
    }

    /// Get all keys (for debugging).
    pub fn keys(&self) -> Vec<Vec<u8>> {
        self.data.keys().cloned().collect()
    }

    /// Get data size.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if FSM is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for KeyValueFSM {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of FSM state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Last included log index.
    pub last_included_index: LogIndex,
    /// Last included log term.
    pub last_included_term: u64,
    /// Serialized state machine data.
    pub data: Vec<u8>,
}

/// Write-Ahead Log for durability.
pub struct WriteAheadLog {
    /// Directory for log files.
    dir: PathBuf,
    /// Current segment file.
    current_segment: Option<Segment>,
    /// Segment metadata.
    segments: Vec<SegmentMeta>,
    /// In-memory index for fast lookups.
    index: BTreeMap<LogIndex, LogPosition>,
    /// All entries (cached).
    entries: Vec<LogEntry>,
}

/// A single log segment file.
struct Segment {
    file: File,
    base_index: LogIndex,
    byte_offset: u64,
}

/// Metadata about a segment.
#[derive(Debug, Clone)]
struct SegmentMeta {
    base_index: LogIndex,
    path: PathBuf,
    size: u64,
}

/// Position of an entry in the log.
#[derive(Debug, Clone, Copy)]
struct LogPosition {
    segment_id: u64,
    offset: u64,
    length: u32,
}

impl WriteAheadLog {
    /// Create a new WAL in the given directory.
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| Error::Storage(e.to_string()))?;

        let mut wal = Self {
            dir,
            current_segment: None,
            segments: Vec::new(),
            index: BTreeMap::new(),
            entries: Vec::new(),
        };

        wal.recover()?;
        Ok(wal)
    }

    /// Recover log from disk.
    fn recover(&mut self) -> Result<()> {
        let mut entries_to_read: Vec<PathBuf> = std::fs::read_dir(&self.dir)
            .map_err(|e| Error::Storage(e.to_string()))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .map(|ext| ext == "log")
                    .unwrap_or(false)
            })
            .collect();

        entries_to_read.sort();

        for path in entries_to_read {
            self.read_segment(&path)?;
        }

        Ok(())
    }

    /// Read entries from a segment file.
    fn read_segment(&mut self, path: &Path) -> Result<()> {
        let mut file = File::open(path).map_err(|e| Error::Storage(e.to_string()))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| Error::Storage(e.to_string()))?;

        let mut offset = 0;
        while offset + 8 <= buffer.len() {
            // Read length
            let length = u32::from_le_bytes([
                buffer[offset],
                buffer[offset + 1],
                buffer[offset + 2],
                buffer[offset + 3],
            ]) as usize;

            // Read CRC
            let stored_crc = u32::from_le_bytes([
                buffer[offset + 4],
                buffer[offset + 5],
                buffer[offset + 6],
                buffer[offset + 7],
            ]);

            if offset + 8 + length > buffer.len() {
                warn!("Truncated entry at offset {}", offset);
                break;
            }

            let data = &buffer[offset + 8..offset + 8 + length];

            // Verify CRC
            let computed_crc = crc32fast::hash(data);
            if computed_crc != stored_crc {
                warn!("CRC mismatch at offset {}", offset);
                break;
            }

            // Deserialize entry
            let entry: LogEntry = bincode::deserialize(data)
                .map_err(|e| Error::Serialization(e.to_string()))?;

            self.entries.push(entry);
            offset += 8 + length;
        }

        Ok(())
    }

    /// Append entries to the log.
    pub fn append(&mut self, entries: &[LogEntry]) -> Result<()> {
        // Ensure we have a current segment
        if self.current_segment.is_none() {
            self.create_new_segment()?;
        }

        let segment = self.current_segment.as_mut().unwrap();

        for entry in entries {
            let data = bincode::serialize(entry)
                .map_err(|e| Error::Serialization(e.to_string()))?;

            // Write: [length: u32][crc: u32][data]
            let crc = crc32fast::hash(&data);
            segment
                .file
                .write_all(&(data.len() as u32).to_le_bytes())
                .map_err(|e| Error::Storage(e.to_string()))?;
            segment
                .file
                .write_all(&crc.to_le_bytes())
                .map_err(|e| Error::Storage(e.to_string()))?;
            segment
                .file
                .write_all(&data)
                .map_err(|e| Error::Storage(e.to_string()))?;

            // Update index
            self.index.insert(
                entry.index,
                LogPosition {
                    segment_id: segment.base_index,
                    offset: segment.byte_offset,
                    length: data.len() as u32,
                },
            );

            segment.byte_offset += 8 + data.len() as u64;
            self.entries.push(entry.clone());
        }

        // Sync to disk
        let segment = self.current_segment.as_mut().unwrap();
        segment
            .file
            .sync_all()
            .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(())
    }

    /// Create a new segment file.
    fn create_new_segment(&mut self) -> Result<()> {
        let base_index = self.entries.last().map(|e| e.index + 1).unwrap_or(1);
        let filename = format!("{:020}.log", base_index);
        let path = self.dir.join(&filename);

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&path)
            .map_err(|e| Error::Storage(e.to_string()))?;

        self.current_segment = Some(Segment {
            file,
            base_index,
            byte_offset: 0,
        });

        self.segments.push(SegmentMeta {
            base_index,
            path,
            size: 0,
        });

        Ok(())
    }

    /// Truncate log from given index.
    pub fn truncate_suffix(&mut self, from_index: LogIndex) -> Result<()> {
        self.entries.retain(|e| e.index < from_index);
        self.index = self.index.split_off(&from_index);

        // In a real implementation, we would also truncate the files
        Ok(())
    }

    /// Compact log up to given index.
    pub fn compact(&mut self, up_to: LogIndex) -> Result<()> {
        self.entries.retain(|e| e.index > up_to);
        let keys_to_remove: Vec<_> = self
            .index
            .range(..=up_to)
            .map(|(k, _)| *k)
            .collect();
        for key in keys_to_remove {
            self.index.remove(&key);
        }

        // In a real implementation, we would also delete old segment files
        Ok(())
    }

    /// Get entries from the log.
    pub fn get_entries(&self, from: LogIndex, to: LogIndex) -> Vec<LogEntry> {
        self.entries
            .iter()
            .filter(|e| e.index >= from && e.index <= to)
            .cloned()
            .collect()
    }

    /// Get last entry.
    pub fn last_entry(&self) -> Option<&LogEntry> {
        self.entries.last()
    }

    /// Get entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Snapshot store for managing snapshots.
pub struct SnapshotStore {
    /// Directory for snapshot files.
    dir: PathBuf,
    /// Current snapshot.
    current: Option<Snapshot>,
}

impl SnapshotStore {
    /// Create a new snapshot store.
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| Error::Storage(e.to_string()))?;

        let mut store = Self {
            dir,
            current: None,
        };

        store.load_latest()?;
        Ok(store)
    }

    /// Load the latest snapshot from disk.
    fn load_latest(&mut self) -> Result<()> {
        let mut snapshots: Vec<PathBuf> = std::fs::read_dir(&self.dir)
            .map_err(|e| Error::Storage(e.to_string()))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .map(|ext| ext == "snap")
                    .unwrap_or(false)
            })
            .collect();

        if snapshots.is_empty() {
            return Ok(());
        }

        snapshots.sort();
        let latest = snapshots.last().unwrap();

        let data = std::fs::read(latest).map_err(|e| Error::Storage(e.to_string()))?;
        let snapshot: Snapshot =
            bincode::deserialize(&data).map_err(|e| Error::Serialization(e.to_string()))?;

        self.current = Some(snapshot);
        Ok(())
    }

    /// Save a snapshot to disk.
    pub fn save(&mut self, snapshot: &Snapshot) -> Result<()> {
        let filename = format!("{:020}.snap", snapshot.last_included_index);
        let path = self.dir.join(&filename);

        let data = bincode::serialize(snapshot).map_err(|e| Error::Serialization(e.to_string()))?;

        std::fs::write(&path, &data).map_err(|e| Error::Storage(e.to_string()))?;

        self.current = Some(snapshot.clone());
        Ok(())
    }

    /// Get current snapshot.
    pub fn get_current(&self) -> Option<&Snapshot> {
        self.current.as_ref()
    }

    /// Write snapshot chunk (for streaming).
    pub fn write_chunk(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        let temp_path = self.dir.join("snapshot.tmp");

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .open(&temp_path)
            .map_err(|e| Error::Storage(e.to_string()))?;

        file.seek(SeekFrom::Start(offset))
            .map_err(|e| Error::Storage(e.to_string()))?;
        file.write_all(data)
            .map_err(|e| Error::Storage(e.to_string()))?;

        Ok(())
    }

    /// Finalize streaming snapshot.
    pub fn finalize(&mut self) -> Result<Snapshot> {
        let temp_path = self.dir.join("snapshot.tmp");
        let data = std::fs::read(&temp_path).map_err(|e| Error::Storage(e.to_string()))?;

        let snapshot: Snapshot =
            bincode::deserialize(&data).map_err(|e| Error::Serialization(e.to_string()))?;

        let filename = format!("{:020}.snap", snapshot.last_included_index);
        let final_path = self.dir.join(&filename);

        std::fs::rename(&temp_path, &final_path).map_err(|e| Error::Storage(e.to_string()))?;

        self.current = Some(snapshot.clone());
        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::EntryType;

    #[test]
    fn test_kvfsm_operations() {
        let mut fsm = KeyValueFSM::new();

        // Put
        let entry = LogEntry {
            term: 1,
            index: 1,
            command: Command::Put {
                key: b"key1".to_vec(),
                value: b"value1".to_vec(),
            },
            entry_type: EntryType::Command,
        };
        fsm.apply(&entry);

        // Get
        assert_eq!(fsm.get(b"key1"), Some(b"value1".to_vec()));

        // Delete
        let entry = LogEntry {
            term: 1,
            index: 2,
            command: Command::Delete {
                key: b"key1".to_vec(),
            },
            entry_type: EntryType::Command,
        };
        fsm.apply(&entry);

        assert_eq!(fsm.get(b"key1"), None);
    }

    #[test]
    fn test_snapshot() {
        let mut fsm = KeyValueFSM::new();

        let entry = LogEntry {
            term: 1,
            index: 1,
            command: Command::Put {
                key: b"key1".to_vec(),
                value: b"value1".to_vec(),
            },
            entry_type: EntryType::Command,
        };
        fsm.apply(&entry);

        let snapshot = fsm.snapshot();
        assert_eq!(snapshot.last_included_index, 1);

        let mut new_fsm = KeyValueFSM::new();
        new_fsm.restore(&snapshot).unwrap();

        assert_eq!(new_fsm.get(b"key1"), Some(b"value1".to_vec()));
    }
}
