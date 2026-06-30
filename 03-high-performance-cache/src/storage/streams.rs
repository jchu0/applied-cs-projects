//! Redis Streams Implementation
//!
//! Streams are an append-only data structure with time-based IDs.
//! This module implements the core stream functionality including:
//! - Stream entries with auto-generated or custom IDs
//! - Consumer groups for distributed processing
//! - Pending entries list (PEL) for acknowledgment tracking
//! - Range queries and trimming

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

/// Stream entry ID in format "timestamp-sequence"
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId {
    /// Milliseconds since Unix epoch
    pub ms: u64,
    /// Sequence number within the millisecond
    pub seq: u64,
}

impl StreamId {
    /// Create a new StreamId
    pub fn new(ms: u64, seq: u64) -> Self {
        Self { ms, seq }
    }

    /// Generate an auto ID based on current time
    pub fn auto(last_id: Option<&StreamId>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        match last_id {
            Some(last) if last.ms >= now => Self::new(last.ms, last.seq + 1),
            Some(last) if last.ms == now => Self::new(now, last.seq + 1),
            _ => Self::new(now, 0),
        }
    }

    /// Parse from string format "ms-seq" or "*"
    pub fn parse(s: &str, last_id: Option<&StreamId>) -> Result<Self, StreamError> {
        if s == "*" {
            return Ok(Self::auto(last_id));
        }

        let parts: Vec<&str> = s.split('-').collect();
        match parts.as_slice() {
            [ms] => {
                let ms: u64 = ms.parse().map_err(|_| StreamError::InvalidId)?;
                Ok(Self::new(ms, 0))
            }
            [ms, "*"] => {
                let ms: u64 = ms.parse().map_err(|_| StreamError::InvalidId)?;
                match last_id {
                    Some(last) if last.ms == ms => Ok(Self::new(ms, last.seq + 1)),
                    _ => Ok(Self::new(ms, 0)),
                }
            }
            [ms, seq] => {
                let ms: u64 = ms.parse().map_err(|_| StreamError::InvalidId)?;
                let seq: u64 = seq.parse().map_err(|_| StreamError::InvalidId)?;
                Ok(Self::new(ms, seq))
            }
            _ => Err(StreamError::InvalidId),
        }
    }

    /// Minimum possible ID
    pub fn min() -> Self {
        Self::new(0, 0)
    }

    /// Maximum possible ID
    pub fn max() -> Self {
        Self::new(u64::MAX, u64::MAX)
    }
}

impl std::fmt::Display for StreamId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.ms, self.seq)
    }
}

/// Stream entry containing ID and field-value pairs
#[derive(Debug, Clone)]
pub struct StreamEntry {
    pub id: StreamId,
    pub fields: Vec<(Vec<u8>, Vec<u8>)>,
}

impl StreamEntry {
    pub fn new(id: StreamId, fields: Vec<(Vec<u8>, Vec<u8>)>) -> Self {
        Self { id, fields }
    }
}

/// Pending entry for consumer tracking
#[derive(Debug, Clone)]
pub struct PendingEntry {
    pub id: StreamId,
    pub consumer: String,
    pub delivery_time: u64,
    pub delivery_count: u32,
}

/// Consumer in a consumer group
#[derive(Debug, Clone)]
pub struct Consumer {
    pub name: String,
    pub last_seen: u64,
    pub pending_count: usize,
}

/// Consumer group for stream processing
#[derive(Debug, Clone)]
pub struct ConsumerGroup {
    pub name: String,
    /// Last delivered ID
    pub last_delivered_id: StreamId,
    /// Pending entries list
    pub pel: BTreeMap<StreamId, PendingEntry>,
    /// Consumers in this group
    pub consumers: HashMap<String, Consumer>,
}

impl ConsumerGroup {
    pub fn new(name: String, start_id: StreamId) -> Self {
        Self {
            name,
            last_delivered_id: start_id,
            pel: BTreeMap::new(),
            consumers: HashMap::new(),
        }
    }

    /// Get or create a consumer
    pub fn get_or_create_consumer(&mut self, name: &str) -> &mut Consumer {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.consumers
            .entry(name.to_string())
            .or_insert_with(|| Consumer {
                name: name.to_string(),
                last_seen: now,
                pending_count: 0,
            })
    }

    /// Acknowledge entries
    pub fn ack(&mut self, ids: &[StreamId]) -> usize {
        let mut count = 0;
        for id in ids {
            if let Some(entry) = self.pel.remove(id) {
                if let Some(consumer) = self.consumers.get_mut(&entry.consumer) {
                    consumer.pending_count = consumer.pending_count.saturating_sub(1);
                }
                count += 1;
            }
        }
        count
    }
}

/// Redis Stream data structure
#[derive(Debug, Clone)]
pub struct Stream {
    /// Stream entries ordered by ID
    entries: BTreeMap<StreamId, StreamEntry>,
    /// Last generated/added ID
    last_id: StreamId,
    /// Consumer groups
    groups: HashMap<String, ConsumerGroup>,
    /// Maximum length (for MAXLEN trimming)
    max_len: Option<usize>,
}

impl Default for Stream {
    fn default() -> Self {
        Self::new()
    }
}

impl Stream {
    /// Create a new empty stream
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            last_id: StreamId::new(0, 0),
            groups: HashMap::new(),
            max_len: None,
        }
    }

    /// Add an entry to the stream
    pub fn add(
        &mut self,
        id_str: &str,
        fields: Vec<(Vec<u8>, Vec<u8>)>,
    ) -> Result<StreamId, StreamError> {
        let id = StreamId::parse(id_str, Some(&self.last_id))?;

        // Ensure ID is greater than last ID
        if id <= self.last_id && !self.entries.is_empty() {
            return Err(StreamError::IdTooSmall);
        }

        let entry = StreamEntry::new(id, fields);
        self.entries.insert(id, entry);
        self.last_id = id;

        // Apply MAXLEN trimming if configured
        if let Some(max_len) = self.max_len {
            self.trim(max_len);
        }

        Ok(id)
    }

    /// Get stream length
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if stream is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the last entry ID
    pub fn last_id(&self) -> StreamId {
        self.last_id
    }

    /// Get the first entry ID
    pub fn first_id(&self) -> Option<StreamId> {
        self.entries.keys().next().copied()
    }

    /// Range query (XRANGE)
    pub fn range(
        &self,
        start: StreamId,
        end: StreamId,
        count: Option<usize>,
    ) -> Vec<&StreamEntry> {
        let mut result: Vec<_> = self
            .entries
            .range(start..=end)
            .map(|(_, entry)| entry)
            .collect();

        if let Some(count) = count {
            result.truncate(count);
        }

        result
    }

    /// Reverse range query (XREVRANGE)
    pub fn revrange(
        &self,
        start: StreamId,
        end: StreamId,
        count: Option<usize>,
    ) -> Vec<&StreamEntry> {
        let mut result: Vec<_> = self
            .entries
            .range(end..=start)
            .rev()
            .map(|(_, entry)| entry)
            .collect();

        if let Some(count) = count {
            result.truncate(count);
        }

        result
    }

    /// Read entries after a given ID (XREAD)
    pub fn read_after(&self, id: StreamId, count: Option<usize>) -> Vec<&StreamEntry> {
        // Find entries with ID > given ID
        let start = StreamId::new(id.ms, id.seq.saturating_add(1));
        self.range(start, StreamId::max(), count)
    }

    /// Trim stream to max length
    pub fn trim(&mut self, max_len: usize) -> usize {
        let current_len = self.entries.len();
        if current_len <= max_len {
            return 0;
        }

        let to_remove = current_len - max_len;
        let keys_to_remove: Vec<_> = self.entries.keys().take(to_remove).copied().collect();

        for key in &keys_to_remove {
            self.entries.remove(key);
        }

        to_remove
    }

    /// Set maximum length for auto-trimming
    pub fn set_max_len(&mut self, max_len: Option<usize>) {
        self.max_len = max_len;
    }

    /// Create a consumer group
    pub fn create_group(&mut self, name: String, start_id: &str) -> Result<(), StreamError> {
        if self.groups.contains_key(&name) {
            return Err(StreamError::GroupExists);
        }

        let id = if start_id == "$" {
            self.last_id
        } else if start_id == "0" {
            StreamId::min()
        } else {
            StreamId::parse(start_id, None)?
        };

        self.groups.insert(name.clone(), ConsumerGroup::new(name, id));
        Ok(())
    }

    /// Get a consumer group
    pub fn get_group(&self, name: &str) -> Option<&ConsumerGroup> {
        self.groups.get(name)
    }

    /// Get a mutable reference to a consumer group
    pub fn get_group_mut(&mut self, name: &str) -> Option<&mut ConsumerGroup> {
        self.groups.get_mut(name)
    }

    /// Destroy a consumer group
    pub fn destroy_group(&mut self, name: &str) -> bool {
        self.groups.remove(name).is_some()
    }

    /// Read from consumer group (XREADGROUP)
    pub fn read_group(
        &mut self,
        group_name: &str,
        consumer_name: &str,
        count: Option<usize>,
        id_spec: &str,
    ) -> Result<Vec<StreamEntry>, StreamError> {
        let group = self
            .groups
            .get_mut(group_name)
            .ok_or(StreamError::NoSuchGroup)?;

        let consumer = group.get_or_create_consumer(consumer_name);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        consumer.last_seen = now;

        // ">" means new entries only
        if id_spec == ">" {
            let start_id = group.last_delivered_id;
            let entries: Vec<StreamEntry> = self
                .entries
                .range((std::ops::Bound::Excluded(start_id), std::ops::Bound::Unbounded))
                .take(count.unwrap_or(usize::MAX))
                .map(|(_, e)| e.clone())
                .collect();

            // Update last delivered and add to PEL
            for entry in &entries {
                group.last_delivered_id = entry.id;
                group.pel.insert(
                    entry.id,
                    PendingEntry {
                        id: entry.id,
                        consumer: consumer_name.to_string(),
                        delivery_time: now,
                        delivery_count: 1,
                    },
                );
            }

            if let Some(consumer) = group.consumers.get_mut(consumer_name) {
                consumer.pending_count += entries.len();
            }

            Ok(entries)
        } else {
            // Return pending entries
            let start_id = StreamId::parse(id_spec, None)?;
            let entries: Vec<StreamEntry> = group
                .pel
                .range(start_id..)
                .filter(|(_, pe)| pe.consumer == consumer_name)
                .take(count.unwrap_or(usize::MAX))
                .filter_map(|(id, pe)| {
                    self.entries.get(id).map(|e| {
                        // Update delivery info
                        StreamEntry {
                            id: pe.id,
                            fields: e.fields.clone(),
                        }
                    })
                })
                .collect();

            // Update delivery count
            for entry in &entries {
                if let Some(pe) = group.pel.get_mut(&entry.id) {
                    pe.delivery_count += 1;
                    pe.delivery_time = now;
                }
            }

            Ok(entries)
        }
    }

    /// Acknowledge entries in a consumer group (XACK)
    pub fn ack(&mut self, group_name: &str, ids: &[StreamId]) -> Result<usize, StreamError> {
        let group = self
            .groups
            .get_mut(group_name)
            .ok_or(StreamError::NoSuchGroup)?;

        Ok(group.ack(ids))
    }

    /// Delete entries from stream (XDEL)
    pub fn delete(&mut self, ids: &[StreamId]) -> usize {
        let mut count = 0;
        for id in ids {
            if self.entries.remove(id).is_some() {
                count += 1;
            }
        }
        count
    }

    /// Get stream info
    pub fn info(&self) -> StreamInfo {
        StreamInfo {
            length: self.entries.len(),
            radix_tree_keys: 1, // Simplified
            radix_tree_nodes: self.entries.len(),
            last_generated_id: self.last_id,
            first_entry: self.entries.values().next().cloned(),
            last_entry: self.entries.values().last().cloned(),
            groups_count: self.groups.len(),
        }
    }

    /// Get all group names
    pub fn group_names(&self) -> Vec<&str> {
        self.groups.keys().map(|s| s.as_str()).collect()
    }
}

/// Stream information
#[derive(Debug, Clone)]
pub struct StreamInfo {
    pub length: usize,
    pub radix_tree_keys: usize,
    pub radix_tree_nodes: usize,
    pub last_generated_id: StreamId,
    pub first_entry: Option<StreamEntry>,
    pub last_entry: Option<StreamEntry>,
    pub groups_count: usize,
}

/// Stream operation errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamError {
    InvalidId,
    IdTooSmall,
    GroupExists,
    NoSuchGroup,
    NoSuchConsumer,
    EmptyFields,
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::InvalidId => write!(f, "ERR Invalid stream ID specified"),
            StreamError::IdTooSmall => {
                write!(f, "ERR The ID specified is equal or smaller than the target stream top item")
            }
            StreamError::GroupExists => write!(f, "BUSYGROUP Consumer Group name already exists"),
            StreamError::NoSuchGroup => write!(f, "NOGROUP No such consumer group"),
            StreamError::NoSuchConsumer => write!(f, "NOGROUP No such consumer"),
            StreamError::EmptyFields => write!(f, "ERR wrong number of arguments for XADD"),
        }
    }
}

impl std::error::Error for StreamError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_id_parse() {
        let id = StreamId::parse("1234-5", None).unwrap();
        assert_eq!(id.ms, 1234);
        assert_eq!(id.seq, 5);

        let id = StreamId::parse("1234", None).unwrap();
        assert_eq!(id.ms, 1234);
        assert_eq!(id.seq, 0);
    }

    #[test]
    fn test_stream_id_auto() {
        let id1 = StreamId::auto(None);
        let id2 = StreamId::auto(Some(&id1));
        assert!(id2 > id1);
    }

    #[test]
    fn test_stream_id_display() {
        let id = StreamId::new(1234, 5);
        assert_eq!(format!("{}", id), "1234-5");
    }

    #[test]
    fn test_stream_add() {
        let mut stream = Stream::new();
        let fields = vec![(b"field1".to_vec(), b"value1".to_vec())];

        let id = stream.add("*", fields.clone()).unwrap();
        assert_eq!(stream.len(), 1);
        assert_eq!(stream.last_id(), id);
    }

    #[test]
    fn test_stream_add_specific_id() {
        let mut stream = Stream::new();
        let fields = vec![(b"field1".to_vec(), b"value1".to_vec())];

        let id = stream.add("1000-0", fields).unwrap();
        assert_eq!(id, StreamId::new(1000, 0));
    }

    #[test]
    fn test_stream_add_id_too_small() {
        let mut stream = Stream::new();
        let fields = vec![(b"field1".to_vec(), b"value1".to_vec())];

        stream.add("1000-0", fields.clone()).unwrap();
        let result = stream.add("500-0", fields);
        assert!(matches!(result, Err(StreamError::IdTooSmall)));
    }

    #[test]
    fn test_stream_range() {
        let mut stream = Stream::new();
        let fields = vec![(b"f".to_vec(), b"v".to_vec())];

        stream.add("1000-0", fields.clone()).unwrap();
        stream.add("2000-0", fields.clone()).unwrap();
        stream.add("3000-0", fields).unwrap();

        let entries = stream.range(StreamId::new(1500, 0), StreamId::new(2500, 0), None);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, StreamId::new(2000, 0));
    }

    #[test]
    fn test_stream_range_with_count() {
        let mut stream = Stream::new();
        let fields = vec![(b"f".to_vec(), b"v".to_vec())];

        for i in 0..10 {
            stream.add(&format!("{}-0", 1000 + i), fields.clone()).unwrap();
        }

        let entries = stream.range(StreamId::min(), StreamId::max(), Some(3));
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn test_stream_read_after() {
        let mut stream = Stream::new();
        let fields = vec![(b"f".to_vec(), b"v".to_vec())];

        stream.add("1000-0", fields.clone()).unwrap();
        stream.add("2000-0", fields.clone()).unwrap();
        stream.add("3000-0", fields).unwrap();

        let entries = stream.read_after(StreamId::new(1000, 0), None);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_stream_trim() {
        let mut stream = Stream::new();
        let fields = vec![(b"f".to_vec(), b"v".to_vec())];

        for i in 0..10 {
            stream.add(&format!("{}-0", 1000 + i), fields.clone()).unwrap();
        }

        let removed = stream.trim(5);
        assert_eq!(removed, 5);
        assert_eq!(stream.len(), 5);
    }

    #[test]
    fn test_stream_delete() {
        let mut stream = Stream::new();
        let fields = vec![(b"f".to_vec(), b"v".to_vec())];

        stream.add("1000-0", fields.clone()).unwrap();
        stream.add("2000-0", fields).unwrap();

        let deleted = stream.delete(&[StreamId::new(1000, 0)]);
        assert_eq!(deleted, 1);
        assert_eq!(stream.len(), 1);
    }

    #[test]
    fn test_consumer_group_create() {
        let mut stream = Stream::new();
        let fields = vec![(b"f".to_vec(), b"v".to_vec())];
        stream.add("1000-0", fields).unwrap();

        stream.create_group("mygroup".to_string(), "0").unwrap();
        assert!(stream.get_group("mygroup").is_some());
    }

    #[test]
    fn test_consumer_group_already_exists() {
        let mut stream = Stream::new();
        stream.create_group("mygroup".to_string(), "0").unwrap();
        let result = stream.create_group("mygroup".to_string(), "0");
        assert!(matches!(result, Err(StreamError::GroupExists)));
    }

    #[test]
    fn test_xreadgroup() {
        let mut stream = Stream::new();
        let fields = vec![(b"f".to_vec(), b"v".to_vec())];

        stream.add("1000-0", fields.clone()).unwrap();
        stream.add("2000-0", fields.clone()).unwrap();
        stream.create_group("mygroup".to_string(), "0").unwrap();

        let entries = stream.read_group("mygroup", "consumer1", Some(1), ">").unwrap();
        assert_eq!(entries.len(), 1);

        let group = stream.get_group("mygroup").unwrap();
        assert_eq!(group.pel.len(), 1);
    }

    #[test]
    fn test_xack() {
        let mut stream = Stream::new();
        let fields = vec![(b"f".to_vec(), b"v".to_vec())];

        let id1 = stream.add("1000-0", fields.clone()).unwrap();
        stream.create_group("mygroup".to_string(), "0").unwrap();

        let entries = stream.read_group("mygroup", "consumer1", None, ">").unwrap();
        assert_eq!(entries.len(), 1);

        let acked = stream.ack("mygroup", &[id1]).unwrap();
        assert_eq!(acked, 1);

        let group = stream.get_group("mygroup").unwrap();
        assert!(group.pel.is_empty());
    }

    #[test]
    fn test_stream_info() {
        let mut stream = Stream::new();
        let fields = vec![(b"f".to_vec(), b"v".to_vec())];

        stream.add("1000-0", fields.clone()).unwrap();
        stream.add("2000-0", fields).unwrap();

        let info = stream.info();
        assert_eq!(info.length, 2);
        assert_eq!(info.last_generated_id, StreamId::new(2000, 0));
    }

    #[test]
    fn test_stream_revrange() {
        let mut stream = Stream::new();
        let fields = vec![(b"f".to_vec(), b"v".to_vec())];

        stream.add("1000-0", fields.clone()).unwrap();
        stream.add("2000-0", fields.clone()).unwrap();
        stream.add("3000-0", fields).unwrap();

        let entries = stream.revrange(StreamId::max(), StreamId::min(), Some(2));
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id, StreamId::new(3000, 0));
        assert_eq!(entries[1].id, StreamId::new(2000, 0));
    }

    #[test]
    fn test_destroy_group() {
        let mut stream = Stream::new();
        stream.create_group("mygroup".to_string(), "0").unwrap();

        assert!(stream.destroy_group("mygroup"));
        assert!(stream.get_group("mygroup").is_none());
    }

    #[test]
    fn test_auto_trim() {
        let mut stream = Stream::new();
        stream.set_max_len(Some(3));

        let fields = vec![(b"f".to_vec(), b"v".to_vec())];
        for i in 0..5 {
            stream.add(&format!("{}-0", 1000 + i), fields.clone()).unwrap();
        }

        assert_eq!(stream.len(), 3);
    }
}
