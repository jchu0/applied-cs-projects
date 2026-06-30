//! Index management for fast message lookups.

use crate::error::{Error, Result};
use bytes::{Buf, BufMut, BytesMut};
use memmap2::{Mmap, MmapMut, MmapOptions};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Entry in the offset index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexEntry {
    /// Relative offset from base.
    pub relative_offset: u32,
    /// Position in the log file.
    pub position: u32,
}

impl IndexEntry {
    /// Size of an index entry in bytes.
    pub const SIZE: usize = 8;

    /// Create a new index entry.
    pub fn new(relative_offset: u32, position: u32) -> Self {
        Self {
            relative_offset,
            position,
        }
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> [u8; 8] {
        let mut buf = [0u8; 8];
        buf[0..4].copy_from_slice(&self.relative_offset.to_be_bytes());
        buf[4..8].copy_from_slice(&self.position.to_be_bytes());
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let relative_offset = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let position = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        Self {
            relative_offset,
            position,
        }
    }
}

/// Offset index for a segment.
pub struct OffsetIndex {
    /// Path to the index file.
    path: PathBuf,
    /// Base offset of the segment.
    base_offset: u64,
    /// Maximum number of entries.
    max_entries: usize,
    /// Current number of entries.
    entries: AtomicU64,
    /// Index file.
    file: RwLock<File>,
    /// Memory-mapped index.
    mmap: RwLock<Option<Mmap>>,
    /// Last offset in the index.
    last_offset: AtomicU64,
}

impl OffsetIndex {
    /// Create a new offset index.
    pub fn new(path: impl Into<PathBuf>, base_offset: u64, max_entries: usize) -> Result<Self> {
        let path = path.into();
        let max_size = max_entries * IndexEntry::SIZE;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;

        // Resize file if needed
        let file_size = file.metadata()?.len() as usize;
        if file_size < max_size {
            file.set_len(max_size as u64)?;
        }

        let entries = (file_size / IndexEntry::SIZE) as u64;

        let mmap = if file_size > 0 {
            Some(unsafe { MmapOptions::new().map(&file)? })
        } else {
            None
        };

        Ok(Self {
            path,
            base_offset,
            max_entries,
            entries: AtomicU64::new(0),
            file: RwLock::new(file),
            mmap: RwLock::new(mmap),
            last_offset: AtomicU64::new(0),
        })
    }

    /// Open an existing offset index.
    pub fn open(path: impl Into<PathBuf>, base_offset: u64) -> Result<Self> {
        let path = path.into();

        if !path.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Index file not found",
            )));
        }

        let file = OpenOptions::new().read(true).write(true).open(&path)?;

        let file_size = file.metadata()?.len() as usize;
        let max_entries = file_size / IndexEntry::SIZE;

        // Count valid entries
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let mut entries = 0u64;
        let mut last_offset = 0u64;

        for i in 0..max_entries {
            let start = i * IndexEntry::SIZE;
            let entry = IndexEntry::from_bytes(&mmap[start..start + IndexEntry::SIZE]);
            if entry.relative_offset == 0 && entry.position == 0 && i > 0 {
                break;
            }
            entries += 1;
            last_offset = base_offset + entry.relative_offset as u64;
        }

        Ok(Self {
            path,
            base_offset,
            max_entries,
            entries: AtomicU64::new(entries),
            file: RwLock::new(file),
            mmap: RwLock::new(Some(mmap)),
            last_offset: AtomicU64::new(last_offset),
        })
    }

    /// Append an entry to the index.
    pub fn append(&self, offset: u64, position: u32) -> Result<()> {
        let entries = self.entries.load(Ordering::Acquire);
        if entries as usize >= self.max_entries {
            return Err(Error::StorageFull);
        }

        let relative_offset = (offset - self.base_offset) as u32;
        let entry = IndexEntry::new(relative_offset, position);

        let pos = entries as u64 * IndexEntry::SIZE as u64;

        let mut file = self.file.write();
        file.seek(SeekFrom::Start(pos))?;
        file.write_all(&entry.to_bytes())?;

        self.entries.fetch_add(1, Ordering::Release);
        self.last_offset.store(offset, Ordering::Release);

        Ok(())
    }

    /// Lookup position for an offset.
    pub fn lookup(&self, offset: u64) -> Option<u32> {
        if offset < self.base_offset {
            return None;
        }

        let relative_offset = (offset - self.base_offset) as u32;
        let entries = self.entries.load(Ordering::Acquire) as usize;

        if entries == 0 {
            return None;
        }

        // Read entries and binary search
        let mmap_guard = self.mmap.read();
        let mmap = mmap_guard.as_ref()?;

        // Binary search
        let mut left = 0;
        let mut right = entries;

        while left < right {
            let mid = left + (right - left) / 2;
            let start = mid * IndexEntry::SIZE;
            let entry = IndexEntry::from_bytes(&mmap[start..start + IndexEntry::SIZE]);

            if entry.relative_offset == relative_offset {
                return Some(entry.position);
            } else if entry.relative_offset < relative_offset {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // Return the position of the largest offset <= target
        if left > 0 {
            let start = (left - 1) * IndexEntry::SIZE;
            let entry = IndexEntry::from_bytes(&mmap[start..start + IndexEntry::SIZE]);
            Some(entry.position)
        } else {
            Some(0)
        }
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.load(Ordering::Acquire) as usize
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get last offset in index.
    pub fn last_offset(&self) -> u64 {
        self.last_offset.load(Ordering::Acquire)
    }

    /// Get base offset.
    pub fn base_offset(&self) -> u64 {
        self.base_offset
    }

    /// Flush to disk.
    pub fn flush(&self) -> Result<()> {
        let file = self.file.write();
        file.sync_all()?;
        Ok(())
    }

    /// Truncate index to a specific offset.
    pub fn truncate_to(&self, offset: u64) -> Result<()> {
        if offset < self.base_offset {
            return Err(Error::OffsetOutOfRange {
                offset,
                start: self.base_offset,
                end: self.last_offset(),
            });
        }

        let relative_offset = (offset - self.base_offset) as u32;
        let entries = self.entries.load(Ordering::Acquire) as usize;

        // Find the entry to truncate to
        let mmap_guard = self.mmap.read();
        if let Some(mmap) = mmap_guard.as_ref() {
            for i in 0..entries {
                let start = i * IndexEntry::SIZE;
                let entry = IndexEntry::from_bytes(&mmap[start..start + IndexEntry::SIZE]);
                if entry.relative_offset > relative_offset {
                    self.entries.store(i as u64, Ordering::Release);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Remap the memory-mapped file.
    pub fn remap(&self) -> Result<()> {
        let file = self.file.read();
        let mmap = unsafe { MmapOptions::new().map(&*file)? };
        *self.mmap.write() = Some(mmap);
        Ok(())
    }
}

/// Time index for timestamp-based lookups.
pub struct TimeIndex {
    /// Path to the index file.
    path: PathBuf,
    /// Base offset.
    base_offset: u64,
    /// Entries: (timestamp, offset).
    entries: RwLock<Vec<(u64, u64)>>,
}

impl TimeIndex {
    /// Entry size in bytes.
    pub const ENTRY_SIZE: usize = 16;

    /// Create a new time index.
    pub fn new(path: impl Into<PathBuf>, base_offset: u64) -> Result<Self> {
        let path = path.into();

        let entries = if path.exists() {
            Self::load_entries(&path)?
        } else {
            Vec::new()
        };

        Ok(Self {
            path,
            base_offset,
            entries: RwLock::new(entries),
        })
    }

    /// Load entries from file.
    fn load_entries(path: &PathBuf) -> Result<Vec<(u64, u64)>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let mut entries = Vec::new();
        let mut cursor = &buffer[..];

        while cursor.len() >= Self::ENTRY_SIZE {
            let timestamp = cursor.get_u64();
            let offset = cursor.get_u64();
            entries.push((timestamp, offset));
        }

        Ok(entries)
    }

    /// Append a timestamp-offset pair.
    pub fn append(&self, timestamp: u64, offset: u64) -> Result<()> {
        let mut entries = self.entries.write();

        // Only append if timestamp is greater than last
        if let Some(&(last_ts, _)) = entries.last() {
            if timestamp <= last_ts {
                return Ok(());
            }
        }

        entries.push((timestamp, offset));

        // Write to file
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)?;

        let mut buf = BytesMut::with_capacity(Self::ENTRY_SIZE);
        buf.put_u64(timestamp);
        buf.put_u64(offset);
        file.write_all(&buf)?;

        Ok(())
    }

    /// Lookup offset for a timestamp.
    pub fn lookup(&self, timestamp: u64) -> Option<u64> {
        let entries = self.entries.read();

        if entries.is_empty() {
            return None;
        }

        // Binary search for the largest timestamp <= target
        let result = entries.binary_search_by(|(ts, _)| ts.cmp(&timestamp));

        match result {
            Ok(idx) => Some(entries[idx].1),
            Err(idx) => {
                if idx > 0 {
                    Some(entries[idx - 1].1)
                } else {
                    Some(entries[0].1)
                }
            }
        }
    }

    /// Get number of entries.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Flush to disk.
    pub fn flush(&self) -> Result<()> {
        // Entries are appended immediately, so nothing to flush
        Ok(())
    }

    /// Truncate to a specific offset.
    pub fn truncate_to(&self, offset: u64) -> Result<()> {
        let mut entries = self.entries.write();
        entries.retain(|(_, off)| *off <= offset);

        // Rewrite file
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)?;

        for (ts, off) in entries.iter() {
            let mut buf = BytesMut::with_capacity(Self::ENTRY_SIZE);
            buf.put_u64(*ts);
            buf.put_u64(*off);
            file.write_all(&buf)?;
        }

        Ok(())
    }
}

/// In-memory index for testing and small datasets.
#[derive(Debug, Default)]
pub struct MemoryIndex {
    /// Offset to position mapping.
    entries: RwLock<BTreeMap<u64, u32>>,
}

impl MemoryIndex {
    /// Create a new memory index.
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(BTreeMap::new()),
        }
    }

    /// Insert an entry.
    pub fn insert(&self, offset: u64, position: u32) {
        self.entries.write().insert(offset, position);
    }

    /// Lookup position for an offset.
    pub fn lookup(&self, offset: u64) -> Option<u32> {
        let entries = self.entries.read();

        // Find the largest key <= offset
        entries
            .range(..=offset)
            .next_back()
            .map(|(_, &pos)| pos)
    }

    /// Get exact position for an offset.
    pub fn get(&self, offset: u64) -> Option<u32> {
        self.entries.read().get(&offset).copied()
    }

    /// Get number of entries.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Clear the index.
    pub fn clear(&self) {
        self.entries.write().clear();
    }

    /// Get all entries.
    pub fn entries(&self) -> Vec<(u64, u32)> {
        self.entries
            .read()
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect()
    }

    /// Remove entries after an offset.
    pub fn truncate_to(&self, offset: u64) {
        let mut entries = self.entries.write();
        entries.split_off(&(offset + 1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_index_entry() {
        let entry = IndexEntry::new(100, 5000);
        let bytes = entry.to_bytes();
        let restored = IndexEntry::from_bytes(&bytes);
        assert_eq!(entry, restored);
    }

    #[test]
    fn test_offset_index_create() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.index");

        let index = OffsetIndex::new(&path, 0, 1000).unwrap();
        assert!(index.is_empty());
        assert_eq!(index.base_offset(), 0);
    }

    #[test]
    fn test_offset_index_append_lookup() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.index");

        let index = OffsetIndex::new(&path, 0, 1000).unwrap();

        index.append(0, 0).unwrap();
        index.append(100, 1000).unwrap();
        index.append(200, 2000).unwrap();
        index.append(300, 3000).unwrap();

        index.remap().unwrap();

        assert_eq!(index.len(), 4);

        // Exact lookups
        assert_eq!(index.lookup(0), Some(0));
        assert_eq!(index.lookup(100), Some(1000));
        assert_eq!(index.lookup(200), Some(2000));
        assert_eq!(index.lookup(300), Some(3000));

        // In-between lookups (should return previous entry's position)
        assert_eq!(index.lookup(50), Some(0));
        assert_eq!(index.lookup(150), Some(1000));
    }

    #[test]
    fn test_memory_index() {
        let index = MemoryIndex::new();

        index.insert(0, 0);
        index.insert(100, 1000);
        index.insert(200, 2000);
        index.insert(300, 3000);

        assert_eq!(index.len(), 4);

        // Exact lookups
        assert_eq!(index.get(100), Some(1000));
        assert_eq!(index.get(150), None);

        // Range lookups
        assert_eq!(index.lookup(0), Some(0));
        assert_eq!(index.lookup(150), Some(1000));
        assert_eq!(index.lookup(350), Some(3000));
    }

    #[test]
    fn test_memory_index_truncate() {
        let index = MemoryIndex::new();

        index.insert(0, 0);
        index.insert(100, 1000);
        index.insert(200, 2000);
        index.insert(300, 3000);

        index.truncate_to(150);

        assert_eq!(index.len(), 2);
        assert!(index.get(200).is_none());
    }

    #[test]
    fn test_time_index() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.timeindex");

        let index = TimeIndex::new(&path, 0).unwrap();

        index.append(1000, 0).unwrap();
        index.append(2000, 100).unwrap();
        index.append(3000, 200).unwrap();

        assert_eq!(index.len(), 3);

        // Exact lookups
        assert_eq!(index.lookup(1000), Some(0));
        assert_eq!(index.lookup(2000), Some(100));

        // In-between lookups
        assert_eq!(index.lookup(1500), Some(0));
        assert_eq!(index.lookup(2500), Some(100));
    }

    #[test]
    fn test_time_index_persistence() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.timeindex");

        // Create and populate
        {
            let index = TimeIndex::new(&path, 0).unwrap();
            index.append(1000, 0).unwrap();
            index.append(2000, 100).unwrap();
        }

        // Reload and verify
        {
            let index = TimeIndex::new(&path, 0).unwrap();
            assert_eq!(index.len(), 2);
            assert_eq!(index.lookup(1000), Some(0));
            assert_eq!(index.lookup(2000), Some(100));
        }
    }

    #[test]
    fn test_offset_index_persistence() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.index");

        // Create and populate
        {
            let index = OffsetIndex::new(&path, 0, 1000).unwrap();
            index.append(0, 0).unwrap();
            index.append(100, 1000).unwrap();
            index.flush().unwrap();
        }

        // Reload and verify
        {
            let index = OffsetIndex::open(&path, 0).unwrap();
            assert_eq!(index.len(), 2);
            assert_eq!(index.lookup(100), Some(1000));
        }
    }

    #[test]
    fn test_memory_index_entries() {
        let index = MemoryIndex::new();
        index.insert(100, 1000);
        index.insert(200, 2000);

        let entries = index.entries();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains(&(100, 1000)));
        assert!(entries.contains(&(200, 2000)));
    }

    #[test]
    fn test_memory_index_clear() {
        let index = MemoryIndex::new();
        index.insert(100, 1000);
        index.insert(200, 2000);

        assert!(!index.is_empty());
        index.clear();
        assert!(index.is_empty());
    }

    #[test]
    fn test_time_index_truncate() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.timeindex");

        let index = TimeIndex::new(&path, 0).unwrap();
        index.append(1000, 0).unwrap();
        index.append(2000, 100).unwrap();
        index.append(3000, 200).unwrap();

        index.truncate_to(100).unwrap();
        assert_eq!(index.len(), 2);
    }
}
