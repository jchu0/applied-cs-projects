//! Log segment management.

use crate::error::{Error, Result};
use crate::index::{MemoryIndex, OffsetIndex, TimeIndex};
use crate::message::{Message, MessageBatch};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use parking_lot::RwLock;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Segment file suffix.
pub const LOG_SUFFIX: &str = ".log";
/// Index file suffix.
pub const INDEX_SUFFIX: &str = ".index";
/// Time index suffix.
pub const TIME_INDEX_SUFFIX: &str = ".timeindex";

/// A segment of the log.
pub struct Segment {
    /// Base offset for this segment.
    base_offset: u64,
    /// Path to the segment directory.
    dir: PathBuf,
    /// Log file.
    log_file: RwLock<File>,
    /// Current size of the log file.
    size: AtomicU64,
    /// Maximum size of the segment.
    max_size: u64,
    /// Offset index.
    offset_index: MemoryIndex,
    /// Time index.
    time_index: RwLock<Vec<(u64, u64)>>,
    /// Next offset to assign.
    next_offset: AtomicU64,
    /// Whether the segment is read-only.
    read_only: AtomicBool,
    /// Index interval (bytes between index entries).
    index_interval: u32,
    /// Bytes since last index entry.
    bytes_since_index: AtomicU64,
}

impl Segment {
    /// Create a new segment.
    pub fn new(dir: impl Into<PathBuf>, base_offset: u64, max_size: u64) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;

        let log_path = dir.join(format!("{:020}{}", base_offset, LOG_SUFFIX));

        let log_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&log_path)?;

        let size = log_file.metadata()?.len();

        Ok(Self {
            base_offset,
            dir,
            log_file: RwLock::new(log_file),
            size: AtomicU64::new(size),
            max_size,
            offset_index: MemoryIndex::new(),
            time_index: RwLock::new(Vec::new()),
            next_offset: AtomicU64::new(base_offset),
            read_only: AtomicBool::new(false),
            index_interval: 4096,
            bytes_since_index: AtomicU64::new(0),
        })
    }

    /// Open an existing segment.
    pub fn open(dir: impl Into<PathBuf>, base_offset: u64, max_size: u64) -> Result<Self> {
        let dir = dir.into();
        let log_path = dir.join(format!("{:020}{}", base_offset, LOG_SUFFIX));

        if !log_path.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Segment not found",
            )));
        }

        let log_file = OpenOptions::new().read(true).write(true).open(&log_path)?;

        let size = log_file.metadata()?.len();

        let segment = Self {
            base_offset,
            dir,
            log_file: RwLock::new(log_file),
            size: AtomicU64::new(size),
            max_size,
            offset_index: MemoryIndex::new(),
            time_index: RwLock::new(Vec::new()),
            next_offset: AtomicU64::new(base_offset),
            read_only: AtomicBool::new(false),
            index_interval: 4096,
            bytes_since_index: AtomicU64::new(0),
        };

        // Rebuild index from log
        segment.rebuild_index()?;

        Ok(segment)
    }

    /// Rebuild index from log file.
    fn rebuild_index(&self) -> Result<()> {
        let mut file = self.log_file.write();
        file.seek(SeekFrom::Start(0))?;

        let mut position = 0u64;
        let mut offset = self.base_offset;
        let mut buffer = [0u8; 8];

        loop {
            // Read message length
            match file.read_exact(&mut buffer[..4]) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }

            let msg_len = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]) as u64;

            // Add to index
            self.offset_index.insert(offset, position as u32);

            // Skip message data
            position += 4 + msg_len;
            file.seek(SeekFrom::Start(position))?;
            offset += 1;
        }

        self.next_offset.store(offset, Ordering::Release);

        Ok(())
    }

    /// Append a message to the segment.
    pub fn append(&self, message: &Message) -> Result<u64> {
        if self.read_only.load(Ordering::Acquire) {
            return Err(Error::IllegalState("Segment is read-only".to_string()));
        }

        let serialized = message.serialize()?;
        let msg_len = serialized.len() as u32;

        let current_size = self.size.load(Ordering::Acquire);
        let new_size = current_size + 4 + msg_len as u64;

        if new_size > self.max_size {
            return Err(Error::StorageFull);
        }

        let offset = self.next_offset.fetch_add(1, Ordering::AcqRel);
        let position = current_size as u32;

        // Write message
        let mut file = self.log_file.write();
        file.seek(SeekFrom::End(0))?;

        let mut buf = BytesMut::with_capacity(4 + serialized.len());
        buf.put_u32(msg_len);
        buf.put_slice(&serialized);
        file.write_all(&buf)?;

        self.size.store(new_size, Ordering::Release);

        // Update index
        let bytes_since = self.bytes_since_index.fetch_add(msg_len as u64 + 4, Ordering::AcqRel);
        if bytes_since >= self.index_interval as u64 {
            self.offset_index.insert(offset, position);
            self.bytes_since_index.store(0, Ordering::Release);
        }

        // Update time index
        {
            let mut time_index = self.time_index.write();
            if time_index.is_empty() || time_index.last().map(|(ts, _)| *ts) != Some(message.timestamp) {
                time_index.push((message.timestamp, offset));
            }
        }

        Ok(offset)
    }

    /// Append a batch of messages.
    pub fn append_batch(&self, batch: &MessageBatch) -> Result<Vec<u64>> {
        let mut offsets = Vec::with_capacity(batch.len());
        for msg in &batch.messages {
            offsets.push(self.append(msg)?);
        }
        Ok(offsets)
    }

    /// Read a message at a specific offset.
    pub fn read(&self, offset: u64) -> Result<Message> {
        let next = self.next_offset.load(Ordering::Acquire);
        if offset < self.base_offset || offset >= next {
            return Err(Error::OffsetOutOfRange {
                offset,
                start: self.base_offset,
                end: next,
            });
        }

        // Find position in index
        let position = self.offset_index.lookup(offset).unwrap_or(0);

        let mut file = self.log_file.write();
        file.seek(SeekFrom::Start(position as u64))?;

        // Scan forward to find the exact offset
        let mut current_offset = self.find_offset_at_position(position)?;
        let mut current_pos = position as u64;

        while current_offset < offset {
            // Read message length
            file.seek(SeekFrom::Start(current_pos))?;
            let mut len_buf = [0u8; 4];
            file.read_exact(&mut len_buf)?;
            let msg_len = u32::from_be_bytes(len_buf) as u64;

            current_pos += 4 + msg_len;
            current_offset += 1;
        }

        // Read the message
        file.seek(SeekFrom::Start(current_pos))?;
        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf)?;
        let msg_len = u32::from_be_bytes(len_buf) as usize;

        let mut msg_buf = vec![0u8; msg_len];
        file.read_exact(&mut msg_buf)?;

        let mut msg = Message::deserialize(&msg_buf)?;
        msg.offset = offset;
        Ok(msg)
    }

    /// Read messages in a range.
    pub fn read_range(&self, start_offset: u64, max_messages: usize) -> Result<Vec<Message>> {
        let mut messages = Vec::with_capacity(max_messages);
        let end_offset = self.next_offset.load(Ordering::Acquire);

        if start_offset >= end_offset {
            return Ok(messages);
        }

        let position = self.offset_index.lookup(start_offset).unwrap_or(0);

        let mut file = self.log_file.write();
        file.seek(SeekFrom::Start(position as u64))?;

        let mut current_offset = self.find_offset_at_position(position)?;
        let mut current_pos = position as u64;

        // Skip to start offset
        while current_offset < start_offset {
            file.seek(SeekFrom::Start(current_pos))?;
            let mut len_buf = [0u8; 4];
            match file.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let msg_len = u32::from_be_bytes(len_buf) as u64;
            current_pos += 4 + msg_len;
            current_offset += 1;
        }

        // Read messages
        while messages.len() < max_messages && current_offset < end_offset {
            file.seek(SeekFrom::Start(current_pos))?;

            let mut len_buf = [0u8; 4];
            match file.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }

            let msg_len = u32::from_be_bytes(len_buf) as usize;
            let mut msg_buf = vec![0u8; msg_len];
            file.read_exact(&mut msg_buf)?;

            let mut msg = Message::deserialize(&msg_buf)?;
            msg.offset = current_offset;
            messages.push(msg);

            current_pos += 4 + msg_len as u64;
            current_offset += 1;
        }

        Ok(messages)
    }

    /// Find offset at a given position.
    fn find_offset_at_position(&self, position: u32) -> Result<u64> {
        // Look up in index to find closest offset
        let entries = self.offset_index.entries();
        for (offset, pos) in entries.iter().rev() {
            if *pos <= position {
                return Ok(*offset);
            }
        }
        Ok(self.base_offset)
    }

    /// Get base offset.
    pub fn base_offset(&self) -> u64 {
        self.base_offset
    }

    /// Get next offset.
    pub fn next_offset(&self) -> u64 {
        self.next_offset.load(Ordering::Acquire)
    }

    /// Get current size.
    pub fn size(&self) -> u64 {
        self.size.load(Ordering::Acquire)
    }

    /// Get maximum size.
    pub fn max_size(&self) -> u64 {
        self.max_size
    }

    /// Check if segment is full.
    pub fn is_full(&self) -> bool {
        self.size.load(Ordering::Acquire) >= self.max_size
    }

    /// Check if segment is read-only.
    pub fn is_read_only(&self) -> bool {
        self.read_only.load(Ordering::Acquire)
    }

    /// Set segment as read-only.
    pub fn set_read_only(&self, read_only: bool) {
        self.read_only.store(read_only, Ordering::Release);
    }

    /// Flush to disk.
    pub fn flush(&self) -> Result<()> {
        let file = self.log_file.write();
        file.sync_all()?;
        Ok(())
    }

    /// Get the log file path.
    pub fn log_path(&self) -> PathBuf {
        self.dir.join(format!("{:020}{}", self.base_offset, LOG_SUFFIX))
    }

    /// Truncate segment to a specific offset.
    pub fn truncate_to(&self, offset: u64) -> Result<()> {
        if offset < self.base_offset {
            return Err(Error::OffsetOutOfRange {
                offset,
                start: self.base_offset,
                end: self.next_offset(),
            });
        }

        // Find position for offset
        let position = if let Some(pos) = self.offset_index.get(offset) {
            pos as u64
        } else {
            // Scan to find position
            let mut file = self.log_file.write();
            file.seek(SeekFrom::Start(0))?;

            let mut current_pos = 0u64;
            let mut current_offset = self.base_offset;

            while current_offset < offset {
                let mut len_buf = [0u8; 4];
                file.read_exact(&mut len_buf)?;
                let msg_len = u32::from_be_bytes(len_buf) as u64;
                current_pos += 4 + msg_len;
                file.seek(SeekFrom::Start(current_pos))?;
                current_offset += 1;
            }

            current_pos
        };

        // Truncate file
        {
            let file = self.log_file.write();
            file.set_len(position)?;
        }

        self.size.store(position, Ordering::Release);
        self.next_offset.store(offset, Ordering::Release);
        self.offset_index.truncate_to(offset);

        Ok(())
    }

    /// Delete the segment files.
    pub fn delete(&self) -> Result<()> {
        let log_path = self.log_path();
        if log_path.exists() {
            std::fs::remove_file(log_path)?;
        }
        Ok(())
    }

    /// Get message count.
    pub fn message_count(&self) -> u64 {
        self.next_offset.load(Ordering::Acquire) - self.base_offset
    }
}

/// Segment manager for handling multiple segments.
pub struct SegmentManager {
    /// Directory for segments.
    dir: PathBuf,
    /// Maximum segment size.
    max_segment_size: u64,
    /// Active segment.
    active_segment: RwLock<Segment>,
    /// Inactive segments.
    segments: RwLock<Vec<Segment>>,
}

impl SegmentManager {
    /// Create a new segment manager.
    pub fn new(dir: impl Into<PathBuf>, max_segment_size: u64) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;

        // Load existing segments
        let mut segment_offsets: Vec<u64> = Vec::new();

        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.ends_with(LOG_SUFFIX) {
                    if let Ok(offset) = name[..20].parse::<u64>() {
                        segment_offsets.push(offset);
                    }
                }
            }
        }

        segment_offsets.sort();

        let (active_segment, inactive_segments) = if segment_offsets.is_empty() {
            // Create first segment
            let segment = Segment::new(&dir, 0, max_segment_size)?;
            (segment, Vec::new())
        } else {
            // Open existing segments
            let mut segments = Vec::new();
            for (i, &offset) in segment_offsets.iter().enumerate() {
                let segment = Segment::open(&dir, offset, max_segment_size)?;
                if i < segment_offsets.len() - 1 {
                    segment.set_read_only(true);
                    segments.push(segment);
                } else {
                    return Ok(Self {
                        dir,
                        max_segment_size,
                        active_segment: RwLock::new(segment),
                        segments: RwLock::new(segments),
                    });
                }
            }

            // Shouldn't reach here, but create new segment if needed
            let last_offset = segment_offsets.last().copied().unwrap_or(0);
            let segment = Segment::new(&dir, last_offset, max_segment_size)?;
            (segment, segments)
        };

        Ok(Self {
            dir,
            max_segment_size,
            active_segment: RwLock::new(active_segment),
            segments: RwLock::new(inactive_segments),
        })
    }

    /// Append a message.
    pub fn append(&self, message: &Message) -> Result<u64> {
        let mut active = self.active_segment.write();

        // Check if we need to roll to a new segment
        if active.is_full() {
            let new_base = active.next_offset();
            active.set_read_only(true);
            let old_segment = std::mem::replace(
                &mut *active,
                Segment::new(&self.dir, new_base, self.max_segment_size)?,
            );
            self.segments.write().push(old_segment);
        }

        match active.append(message) {
            Ok(offset) => Ok(offset),
            Err(Error::StorageFull) => {
                // Segment can't fit this message; roll to a new segment and retry
                let new_base = active.next_offset();
                active.set_read_only(true);
                let old_segment = std::mem::replace(
                    &mut *active,
                    Segment::new(&self.dir, new_base, self.max_segment_size)?,
                );
                self.segments.write().push(old_segment);
                active.append(message)
            }
            Err(e) => Err(e),
        }
    }

    /// Read a message.
    pub fn read(&self, offset: u64) -> Result<Message> {
        // Check active segment first
        {
            let active = self.active_segment.read();
            if offset >= active.base_offset() {
                return active.read(offset);
            }
        }

        // Search inactive segments
        let segments = self.segments.read();
        for segment in segments.iter().rev() {
            if offset >= segment.base_offset() && offset < segment.next_offset() {
                return segment.read(offset);
            }
        }

        Err(Error::OffsetOutOfRange {
            offset,
            start: self.start_offset(),
            end: self.end_offset(),
        })
    }

    /// Read messages in a range.
    pub fn read_range(&self, start_offset: u64, max_messages: usize) -> Result<Vec<Message>> {
        let mut messages = Vec::new();
        let mut remaining = max_messages;
        let mut current_offset = start_offset;

        // Read from inactive segments
        {
            let segments = self.segments.read();
            for segment in segments.iter() {
                if remaining == 0 {
                    break;
                }
                if current_offset >= segment.next_offset() {
                    continue;
                }
                if current_offset >= segment.base_offset() {
                    let batch = segment.read_range(current_offset, remaining)?;
                    remaining -= batch.len();
                    if let Some(last) = batch.last() {
                        current_offset = last.offset + 1;
                    }
                    messages.extend(batch);
                }
            }
        }

        // Read from active segment
        if remaining > 0 {
            let active = self.active_segment.read();
            if current_offset >= active.base_offset() {
                let batch = active.read_range(current_offset, remaining)?;
                messages.extend(batch);
            }
        }

        Ok(messages)
    }

    /// Get start offset (earliest available).
    pub fn start_offset(&self) -> u64 {
        let segments = self.segments.read();
        if let Some(first) = segments.first() {
            first.base_offset()
        } else {
            self.active_segment.read().base_offset()
        }
    }

    /// Get end offset (next to be written).
    pub fn end_offset(&self) -> u64 {
        self.active_segment.read().next_offset()
    }

    /// Get high watermark (last committed offset).
    pub fn high_watermark(&self) -> u64 {
        let end = self.end_offset();
        if end > 0 {
            end - 1
        } else {
            0
        }
    }

    /// Flush all segments.
    pub fn flush(&self) -> Result<()> {
        for segment in self.segments.read().iter() {
            segment.flush()?;
        }
        self.active_segment.read().flush()?;
        Ok(())
    }

    /// Delete old segments based on retention.
    pub fn delete_before(&self, offset: u64) -> Result<usize> {
        let mut segments = self.segments.write();
        let mut deleted = 0;

        segments.retain(|segment| {
            if segment.next_offset() <= offset {
                let _ = segment.delete();
                deleted += 1;
                false
            } else {
                true
            }
        });

        Ok(deleted)
    }

    /// Get total size.
    pub fn total_size(&self) -> u64 {
        let segments = self.segments.read();
        let inactive_size: u64 = segments.iter().map(|s| s.size()).sum();
        inactive_size + self.active_segment.read().size()
    }

    /// Get segment count.
    pub fn segment_count(&self) -> usize {
        self.segments.read().len() + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_message(payload: &str) -> Message {
        Message::new(payload.as_bytes().to_vec())
    }

    #[test]
    fn test_segment_create() {
        let dir = TempDir::new().unwrap();
        let segment = Segment::new(dir.path(), 0, 1024 * 1024).unwrap();

        assert_eq!(segment.base_offset(), 0);
        assert_eq!(segment.next_offset(), 0);
        assert_eq!(segment.size(), 0);
        assert!(!segment.is_full());
    }

    #[test]
    fn test_segment_append_read() {
        let dir = TempDir::new().unwrap();
        let segment = Segment::new(dir.path(), 0, 1024 * 1024).unwrap();

        let msg1 = create_test_message("Hello");
        let msg2 = create_test_message("World");

        let offset1 = segment.append(&msg1).unwrap();
        let offset2 = segment.append(&msg2).unwrap();

        assert_eq!(offset1, 0);
        assert_eq!(offset2, 1);

        let read1 = segment.read(0).unwrap();
        let read2 = segment.read(1).unwrap();

        assert_eq!(read1.payload, msg1.payload);
        assert_eq!(read2.payload, msg2.payload);
    }

    #[test]
    fn test_segment_read_range() {
        let dir = TempDir::new().unwrap();
        let segment = Segment::new(dir.path(), 0, 1024 * 1024).unwrap();

        for i in 0..10 {
            let msg = create_test_message(&format!("Message {}", i));
            segment.append(&msg).unwrap();
        }

        let messages = segment.read_range(3, 5).unwrap();
        assert_eq!(messages.len(), 5);

        for (i, msg) in messages.iter().enumerate() {
            let expected = format!("Message {}", i + 3);
            assert_eq!(msg.payload.as_ref(), expected.as_bytes());
        }
    }

    #[test]
    fn test_segment_persistence() {
        let dir = TempDir::new().unwrap();

        // Create and write
        {
            let segment = Segment::new(dir.path(), 0, 1024 * 1024).unwrap();
            segment.append(&create_test_message("Test 1")).unwrap();
            segment.append(&create_test_message("Test 2")).unwrap();
            segment.flush().unwrap();
        }

        // Reopen and read
        {
            let segment = Segment::open(dir.path(), 0, 1024 * 1024).unwrap();
            assert_eq!(segment.next_offset(), 2);

            let msg = segment.read(0).unwrap();
            assert_eq!(msg.payload.as_ref(), b"Test 1");
        }
    }

    #[test]
    fn test_segment_full() {
        let dir = TempDir::new().unwrap();
        let segment = Segment::new(dir.path(), 0, 100).unwrap(); // Very small segment

        // Keep appending until full
        let mut count = 0;
        for _ in 0..100 {
            match segment.append(&create_test_message("X")) {
                Ok(_) => count += 1,
                Err(Error::StorageFull) => break,
                Err(e) => panic!("Unexpected error: {}", e),
            }
        }

        assert!(count > 0);
        // StorageFull was returned, meaning the segment can't fit another message.
        // The stored size may be less than max_size since the last message didn't fit.
        assert!(count < 100);
    }

    #[test]
    fn test_segment_manager() {
        let dir = TempDir::new().unwrap();
        let manager = SegmentManager::new(dir.path(), 1024 * 1024).unwrap();

        let msg1 = create_test_message("Hello");
        let msg2 = create_test_message("World");

        let offset1 = manager.append(&msg1).unwrap();
        let offset2 = manager.append(&msg2).unwrap();

        assert_eq!(offset1, 0);
        assert_eq!(offset2, 1);

        let read1 = manager.read(0).unwrap();
        let read2 = manager.read(1).unwrap();

        assert_eq!(read1.payload, msg1.payload);
        assert_eq!(read2.payload, msg2.payload);
    }

    #[test]
    fn test_segment_manager_rolling() {
        let dir = TempDir::new().unwrap();
        let manager = SegmentManager::new(dir.path(), 200).unwrap(); // Small segments

        // Write enough messages to cause segment roll
        for i in 0..50 {
            let msg = create_test_message(&format!("Message {}", i));
            manager.append(&msg).unwrap();
        }

        assert!(manager.segment_count() > 1);

        // Verify all messages are readable
        let messages = manager.read_range(0, 50).unwrap();
        assert_eq!(messages.len(), 50);
    }

    #[test]
    fn test_segment_truncate() {
        let dir = TempDir::new().unwrap();
        let segment = Segment::new(dir.path(), 0, 1024 * 1024).unwrap();

        for i in 0..10 {
            segment.append(&create_test_message(&format!("Msg {}", i))).unwrap();
        }

        assert_eq!(segment.next_offset(), 10);

        segment.truncate_to(5).unwrap();
        assert_eq!(segment.next_offset(), 5);

        // Reading offset 5+ should fail
        assert!(segment.read(5).is_err());
    }

    #[test]
    fn test_segment_manager_offsets() {
        let dir = TempDir::new().unwrap();
        let manager = SegmentManager::new(dir.path(), 1024 * 1024).unwrap();

        assert_eq!(manager.start_offset(), 0);
        assert_eq!(manager.end_offset(), 0);

        manager.append(&create_test_message("Test")).unwrap();

        assert_eq!(manager.end_offset(), 1);
        assert_eq!(manager.high_watermark(), 0);
    }

    #[test]
    fn test_segment_read_only() {
        let dir = TempDir::new().unwrap();
        let segment = Segment::new(dir.path(), 0, 1024 * 1024).unwrap();

        segment.append(&create_test_message("Test")).unwrap();
        segment.set_read_only(true);

        assert!(segment.is_read_only());
        assert!(segment.append(&create_test_message("Fail")).is_err());
    }

    #[test]
    fn test_segment_message_count() {
        let dir = TempDir::new().unwrap();
        let segment = Segment::new(dir.path(), 0, 1024 * 1024).unwrap();

        assert_eq!(segment.message_count(), 0);

        for _ in 0..5 {
            segment.append(&create_test_message("Test")).unwrap();
        }

        assert_eq!(segment.message_count(), 5);
    }

    #[test]
    fn test_segment_manager_total_size() {
        let dir = TempDir::new().unwrap();
        let manager = SegmentManager::new(dir.path(), 1024 * 1024).unwrap();

        assert_eq!(manager.total_size(), 0);

        manager.append(&create_test_message("Hello World")).unwrap();

        assert!(manager.total_size() > 0);
    }
}
