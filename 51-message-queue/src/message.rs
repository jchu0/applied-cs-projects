//! Message types and serialization.

use crate::compression::Compression;
use crate::error::{Error, Result};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Unique message identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(pub Uuid);

impl MessageId {
    /// Generate a new random message ID.
    pub fn new() -> Self {
        MessageId(Uuid::new_v4())
    }

    /// Create from UUID.
    pub fn from_uuid(uuid: Uuid) -> Self {
        MessageId(uuid)
    }

    /// Create from bytes (must be 16 bytes).
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 16 {
            return None;
        }
        let uuid = Uuid::from_slice(bytes).ok()?;
        Some(MessageId(uuid))
    }

    /// Get as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }

    /// Get the underlying UUID.
    pub fn uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for MessageId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for MessageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for MessageId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(MessageId(Uuid::parse_str(s)?))
    }
}

/// Message headers (key-value pairs).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Headers {
    inner: HashMap<String, Vec<u8>>,
}

impl Headers {
    /// Create empty headers.
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Create headers with initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: HashMap::with_capacity(capacity),
        }
    }

    /// Insert a string header.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Vec<u8>>) {
        self.inner.insert(key.into(), value.into());
    }

    /// Insert a string value.
    pub fn insert_string(&mut self, key: impl Into<String>, value: impl AsRef<str>) {
        self.inner
            .insert(key.into(), value.as_ref().as_bytes().to_vec());
    }

    /// Get a header value.
    pub fn get(&self, key: &str) -> Option<&[u8]> {
        self.inner.get(key).map(|v| v.as_slice())
    }

    /// Get a header as string.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.inner
            .get(key)
            .and_then(|v| std::str::from_utf8(v).ok())
    }

    /// Remove a header.
    pub fn remove(&mut self, key: &str) -> Option<Vec<u8>> {
        self.inner.remove(key)
    }

    /// Check if header exists.
    pub fn contains(&self, key: &str) -> bool {
        self.inner.contains_key(key)
    }

    /// Get number of headers.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterate over headers.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Vec<u8>)> {
        self.inner.iter()
    }

    /// Get the inner HashMap.
    pub fn into_inner(self) -> HashMap<String, Vec<u8>> {
        self.inner
    }
}

impl From<HashMap<String, Vec<u8>>> for Headers {
    fn from(map: HashMap<String, Vec<u8>>) -> Self {
        Headers { inner: map }
    }
}

impl FromIterator<(String, Vec<u8>)> for Headers {
    fn from_iter<T: IntoIterator<Item = (String, Vec<u8>)>>(iter: T) -> Self {
        Headers {
            inner: iter.into_iter().collect(),
        }
    }
}

/// A message in the queue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Unique message ID.
    pub id: MessageId,
    /// Optional message key (used for partitioning).
    pub key: Option<Bytes>,
    /// Message payload.
    pub payload: Bytes,
    /// Message headers.
    pub headers: Headers,
    /// Timestamp (milliseconds since epoch).
    pub timestamp: u64,
    /// Offset within partition (set when message is stored).
    pub offset: u64,
    /// Partition this message belongs to (set when message is stored).
    pub partition: u32,
}

impl Message {
    /// Create a new message with payload.
    pub fn new(payload: impl Into<Bytes>) -> Self {
        Self {
            id: MessageId::new(),
            key: None,
            payload: payload.into(),
            headers: Headers::new(),
            timestamp: current_timestamp(),
            offset: 0,
            partition: 0,
        }
    }

    /// Create a new message with key and payload.
    pub fn with_key(key: impl Into<Bytes>, payload: impl Into<Bytes>) -> Self {
        Self {
            id: MessageId::new(),
            key: Some(key.into()),
            payload: payload.into(),
            headers: Headers::new(),
            timestamp: current_timestamp(),
            offset: 0,
            partition: 0,
        }
    }

    /// Set the message key.
    pub fn set_key(mut self, key: impl Into<Bytes>) -> Self {
        self.key = Some(key.into());
        self
    }

    /// Add a header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<Vec<u8>>) -> Self {
        self.headers.insert(key, value);
        self
    }

    /// Set timestamp.
    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Get size in bytes (approximate).
    pub fn size(&self) -> usize {
        16 // message id
        + self.key.as_ref().map(|k| k.len()).unwrap_or(0) + 4 // key + length
        + self.payload.len() + 4 // payload + length
        + self.headers_size() + 4 // headers + count
        + 8 // timestamp
        + 8 // offset
        + 4 // partition
    }

    /// Get headers size.
    fn headers_size(&self) -> usize {
        self.headers
            .iter()
            .map(|(k, v)| k.len() + 4 + v.len() + 4)
            .sum()
    }

    /// Serialize message to bytes.
    pub fn serialize(&self) -> Result<Bytes> {
        let mut buf = BytesMut::with_capacity(self.size());

        // Magic byte + version
        buf.put_u8(0x01);

        // Message ID (16 bytes)
        buf.put_slice(self.id.as_bytes());

        // Key (optional)
        if let Some(ref key) = self.key {
            buf.put_i32(key.len() as i32);
            buf.put_slice(key);
        } else {
            buf.put_i32(-1);
        }

        // Payload
        buf.put_i32(self.payload.len() as i32);
        buf.put_slice(&self.payload);

        // Headers count
        buf.put_i32(self.headers.len() as i32);
        for (k, v) in self.headers.iter() {
            buf.put_i32(k.len() as i32);
            buf.put_slice(k.as_bytes());
            buf.put_i32(v.len() as i32);
            buf.put_slice(v);
        }

        // Timestamp
        buf.put_u64(self.timestamp);

        // Offset
        buf.put_u64(self.offset);

        // Partition
        buf.put_u32(self.partition);

        // CRC32 checksum
        let crc = crc32fast::hash(&buf);
        buf.put_u32(crc);

        Ok(buf.freeze())
    }

    /// Deserialize message from bytes.
    pub fn deserialize(mut data: &[u8]) -> Result<Self> {
        if data.len() < 5 {
            return Err(Error::Serialization("Message too short".to_string()));
        }

        // Verify CRC
        let crc_pos = data.len() - 4;
        let stored_crc = u32::from_be_bytes([
            data[crc_pos],
            data[crc_pos + 1],
            data[crc_pos + 2],
            data[crc_pos + 3],
        ]);
        let calculated_crc = crc32fast::hash(&data[..crc_pos]);
        if stored_crc != calculated_crc {
            return Err(Error::CrcMismatch {
                expected: stored_crc,
                actual: calculated_crc,
            });
        }

        // Magic byte
        let magic = data.get_u8();
        if magic != 0x01 {
            return Err(Error::Serialization(format!(
                "Invalid magic byte: {}",
                magic
            )));
        }

        // Message ID
        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(&data[..16]);
        data.advance(16);
        let id = MessageId::from_bytes(&id_bytes).ok_or_else(|| {
            Error::Serialization("Invalid message ID".to_string())
        })?;

        // Key
        let key_len = data.get_i32();
        let key = if key_len >= 0 {
            let len = key_len as usize;
            let key = Bytes::copy_from_slice(&data[..len]);
            data.advance(len);
            Some(key)
        } else {
            None
        };

        // Payload
        let payload_len = data.get_i32() as usize;
        let payload = Bytes::copy_from_slice(&data[..payload_len]);
        data.advance(payload_len);

        // Headers
        let header_count = data.get_i32() as usize;
        let mut headers = Headers::with_capacity(header_count);
        for _ in 0..header_count {
            let key_len = data.get_i32() as usize;
            let key = std::str::from_utf8(&data[..key_len])
                .map_err(|e| Error::Serialization(e.to_string()))?
                .to_string();
            data.advance(key_len);

            let val_len = data.get_i32() as usize;
            let value = data[..val_len].to_vec();
            data.advance(val_len);

            headers.insert(key, value);
        }

        // Timestamp
        let timestamp = data.get_u64();

        // Offset
        let offset = data.get_u64();

        // Partition
        let partition = data.get_u32();

        Ok(Message {
            id,
            key,
            payload,
            headers,
            timestamp,
            offset,
            partition,
        })
    }
}

/// A batch of messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageBatch {
    /// Messages in the batch.
    pub messages: Vec<Message>,
    /// Compression used for the batch.
    pub compression: Compression,
    /// First offset in the batch.
    pub base_offset: u64,
    /// Partition ID.
    pub partition: u32,
    /// Producer ID (for idempotent producers).
    pub producer_id: Option<u64>,
    /// Sequence number (for idempotent producers).
    pub sequence: Option<u32>,
}

impl MessageBatch {
    /// Create a new empty batch.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            compression: Compression::None,
            base_offset: 0,
            partition: 0,
            producer_id: None,
            sequence: None,
        }
    }

    /// Create a batch with messages.
    pub fn with_messages(messages: Vec<Message>) -> Self {
        Self {
            messages,
            compression: Compression::None,
            base_offset: 0,
            partition: 0,
            producer_id: None,
            sequence: None,
        }
    }

    /// Set compression.
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Set base offset.
    pub fn with_base_offset(mut self, offset: u64) -> Self {
        self.base_offset = offset;
        self
    }

    /// Set partition.
    pub fn with_partition(mut self, partition: u32) -> Self {
        self.partition = partition;
        self
    }

    /// Add a message to the batch.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Get number of messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Get total size of messages.
    pub fn size(&self) -> usize {
        self.messages.iter().map(|m| m.size()).sum()
    }

    /// Serialize the batch to bytes.
    pub fn serialize(&self) -> Result<Bytes> {
        let mut buf = BytesMut::new();

        // Magic byte
        buf.put_u8(0x02);

        // Compression type
        buf.put_u8(self.compression.as_u8());

        // Base offset
        buf.put_u64(self.base_offset);

        // Partition
        buf.put_u32(self.partition);

        // Producer ID
        if let Some(pid) = self.producer_id {
            buf.put_u8(1);
            buf.put_u64(pid);
        } else {
            buf.put_u8(0);
        }

        // Sequence
        if let Some(seq) = self.sequence {
            buf.put_u8(1);
            buf.put_u32(seq);
        } else {
            buf.put_u8(0);
        }

        // Serialize messages
        let mut messages_buf = BytesMut::new();
        messages_buf.put_u32(self.messages.len() as u32);
        for msg in &self.messages {
            let serialized = msg.serialize()?;
            messages_buf.put_u32(serialized.len() as u32);
            messages_buf.put_slice(&serialized);
        }

        // Compress messages if needed
        let compressed = self.compression.compress(&messages_buf)?;
        buf.put_u32(compressed.len() as u32);
        buf.put_slice(&compressed);

        // CRC32 checksum
        let crc = crc32fast::hash(&buf);
        buf.put_u32(crc);

        Ok(buf.freeze())
    }

    /// Deserialize batch from bytes.
    pub fn deserialize(mut data: &[u8]) -> Result<Self> {
        if data.len() < 10 {
            return Err(Error::Serialization("Batch too short".to_string()));
        }

        // Verify CRC
        let crc_pos = data.len() - 4;
        let stored_crc = u32::from_be_bytes([
            data[crc_pos],
            data[crc_pos + 1],
            data[crc_pos + 2],
            data[crc_pos + 3],
        ]);
        let calculated_crc = crc32fast::hash(&data[..crc_pos]);
        if stored_crc != calculated_crc {
            return Err(Error::CrcMismatch {
                expected: stored_crc,
                actual: calculated_crc,
            });
        }

        // Magic byte
        let magic = data.get_u8();
        if magic != 0x02 {
            return Err(Error::Serialization(format!(
                "Invalid batch magic byte: {}",
                magic
            )));
        }

        // Compression
        let compression = Compression::from_u8(data.get_u8())
            .ok_or_else(|| Error::Serialization("Invalid compression type".to_string()))?;

        // Base offset
        let base_offset = data.get_u64();

        // Partition
        let partition = data.get_u32();

        // Producer ID
        let producer_id = if data.get_u8() == 1 {
            Some(data.get_u64())
        } else {
            None
        };

        // Sequence
        let sequence = if data.get_u8() == 1 {
            Some(data.get_u32())
        } else {
            None
        };

        // Compressed messages length
        let compressed_len = data.get_u32() as usize;
        let compressed_data = &data[..compressed_len];
        data.advance(compressed_len);

        // Decompress
        let decompressed = compression.decompress(compressed_data)?;

        // Parse messages
        let mut msg_data = decompressed.as_ref();
        let msg_count = msg_data.get_u32() as usize;
        let mut messages = Vec::with_capacity(msg_count);

        for _ in 0..msg_count {
            let msg_len = msg_data.get_u32() as usize;
            let msg = Message::deserialize(&msg_data[..msg_len])?;
            messages.push(msg);
            msg_data.advance(msg_len);
        }

        Ok(MessageBatch {
            messages,
            compression,
            base_offset,
            partition,
            producer_id,
            sequence,
        })
    }

    /// Iterate over messages.
    pub fn iter(&self) -> impl Iterator<Item = &Message> {
        self.messages.iter()
    }
}

impl Default for MessageBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl IntoIterator for MessageBatch {
    type Item = Message;
    type IntoIter = std::vec::IntoIter<Message>;

    fn into_iter(self) -> Self::IntoIter {
        self.messages.into_iter()
    }
}

/// Get current timestamp in milliseconds.
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Message builder for fluent construction.
#[derive(Debug, Clone)]
pub struct MessageBuilder {
    key: Option<Bytes>,
    payload: Bytes,
    headers: Headers,
    timestamp: Option<u64>,
}

impl MessageBuilder {
    /// Create a new message builder with payload.
    pub fn new(payload: impl Into<Bytes>) -> Self {
        Self {
            key: None,
            payload: payload.into(),
            headers: Headers::new(),
            timestamp: None,
        }
    }

    /// Set the message key.
    pub fn key(mut self, key: impl Into<Bytes>) -> Self {
        self.key = Some(key.into());
        self
    }

    /// Add a header.
    pub fn header(mut self, key: impl Into<String>, value: impl Into<Vec<u8>>) -> Self {
        self.headers.insert(key, value);
        self
    }

    /// Add a string header.
    pub fn string_header(mut self, key: impl Into<String>, value: impl AsRef<str>) -> Self {
        self.headers.insert_string(key, value);
        self
    }

    /// Set timestamp.
    pub fn timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Build the message.
    pub fn build(self) -> Message {
        Message {
            id: MessageId::new(),
            key: self.key,
            payload: self.payload,
            headers: self.headers,
            timestamp: self.timestamp.unwrap_or_else(current_timestamp),
            offset: 0,
            partition: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_id_new() {
        let id1 = MessageId::new();
        let id2 = MessageId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_message_id_from_bytes() {
        let id = MessageId::new();
        let bytes = id.as_bytes();
        let id2 = MessageId::from_bytes(bytes).unwrap();
        assert_eq!(id, id2);
    }

    #[test]
    fn test_message_id_parse() {
        let id = MessageId::new();
        let s = id.to_string();
        let id2: MessageId = s.parse().unwrap();
        assert_eq!(id, id2);
    }

    #[test]
    fn test_headers() {
        let mut headers = Headers::new();
        headers.insert("key1", vec![1, 2, 3]);
        headers.insert_string("key2", "value2");

        assert_eq!(headers.get("key1"), Some([1, 2, 3].as_slice()));
        assert_eq!(headers.get_string("key2"), Some("value2"));
        assert_eq!(headers.len(), 2);
        assert!(!headers.is_empty());
        assert!(headers.contains("key1"));
        assert!(!headers.contains("key3"));
    }

    #[test]
    fn test_message_new() {
        let msg = Message::new("Hello, World!");
        assert_eq!(msg.payload.as_ref(), b"Hello, World!");
        assert!(msg.key.is_none());
        assert!(msg.headers.is_empty());
    }

    #[test]
    fn test_message_with_key() {
        let msg = Message::with_key("my-key", "Hello, World!");
        assert_eq!(msg.key.as_ref().unwrap().as_ref(), b"my-key");
        assert_eq!(msg.payload.as_ref(), b"Hello, World!");
    }

    #[test]
    fn test_message_builder() {
        let msg = MessageBuilder::new("payload")
            .key("key")
            .header("h1", vec![1, 2, 3])
            .string_header("h2", "value")
            .timestamp(12345)
            .build();

        assert_eq!(msg.key.as_ref().unwrap().as_ref(), b"key");
        assert_eq!(msg.payload.as_ref(), b"payload");
        assert_eq!(msg.timestamp, 12345);
        assert_eq!(msg.headers.len(), 2);
    }

    #[test]
    fn test_message_serialize_deserialize() {
        let msg = MessageBuilder::new("Hello, World!")
            .key("test-key")
            .header("content-type", b"text/plain".to_vec())
            .timestamp(1234567890)
            .build();

        let serialized = msg.serialize().unwrap();
        let deserialized = Message::deserialize(&serialized).unwrap();

        assert_eq!(msg.id, deserialized.id);
        assert_eq!(msg.key, deserialized.key);
        assert_eq!(msg.payload, deserialized.payload);
        assert_eq!(msg.headers, deserialized.headers);
        assert_eq!(msg.timestamp, deserialized.timestamp);
    }

    #[test]
    fn test_message_without_key() {
        let msg = Message::new("payload");
        let serialized = msg.serialize().unwrap();
        let deserialized = Message::deserialize(&serialized).unwrap();

        assert_eq!(msg.id, deserialized.id);
        assert!(deserialized.key.is_none());
    }

    #[test]
    fn test_message_size() {
        let msg = Message::with_key("key", "Hello, World!")
            .with_header("h1", vec![1, 2, 3]);
        assert!(msg.size() > 0);
    }

    #[test]
    fn test_message_batch() {
        let mut batch = MessageBatch::new();
        batch.push(Message::new("msg1"));
        batch.push(Message::new("msg2"));
        batch.push(Message::new("msg3"));

        assert_eq!(batch.len(), 3);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_message_batch_serialize() {
        let batch = MessageBatch::with_messages(vec![
            Message::new("msg1"),
            Message::new("msg2"),
        ])
        .with_compression(Compression::None)
        .with_base_offset(100)
        .with_partition(1);

        let serialized = batch.serialize().unwrap();
        let deserialized = MessageBatch::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.len(), 2);
        assert_eq!(deserialized.base_offset, 100);
        assert_eq!(deserialized.partition, 1);
    }

    #[test]
    fn test_message_batch_with_compression() {
        let batch = MessageBatch::with_messages(vec![
            Message::new("This is a longer message for better compression"),
            Message::new("Another message with some content"),
        ])
        .with_compression(Compression::Lz4);

        let serialized = batch.serialize().unwrap();
        let deserialized = MessageBatch::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.len(), 2);
        assert_eq!(deserialized.compression, Compression::Lz4);
    }

    #[test]
    fn test_message_crc_validation() {
        let msg = Message::new("Hello");
        let mut serialized = msg.serialize().unwrap().to_vec();

        // Corrupt the data
        serialized[10] ^= 0xFF;

        let result = Message::deserialize(&serialized);
        assert!(matches!(result, Err(Error::CrcMismatch { .. })));
    }

    #[test]
    fn test_current_timestamp() {
        let ts = current_timestamp();
        assert!(ts > 0);
    }

    #[test]
    fn test_empty_batch() {
        let batch = MessageBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);

        let serialized = batch.serialize().unwrap();
        let deserialized = MessageBatch::deserialize(&serialized).unwrap();
        assert!(deserialized.is_empty());
    }

    #[test]
    fn test_headers_from_hashmap() {
        let mut map = HashMap::new();
        map.insert("key1".to_string(), vec![1, 2, 3]);
        map.insert("key2".to_string(), b"value".to_vec());

        let headers: Headers = map.into();
        assert_eq!(headers.len(), 2);
    }

    #[test]
    fn test_headers_iter() {
        let mut headers = Headers::new();
        headers.insert("k1", vec![1]);
        headers.insert("k2", vec![2]);

        let count = headers.iter().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_message_batch_iter() {
        let batch = MessageBatch::with_messages(vec![
            Message::new("msg1"),
            Message::new("msg2"),
        ]);

        let messages: Vec<_> = batch.iter().collect();
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_message_batch_into_iter() {
        let batch = MessageBatch::with_messages(vec![
            Message::new("msg1"),
            Message::new("msg2"),
        ]);

        let messages: Vec<_> = batch.into_iter().collect();
        assert_eq!(messages.len(), 2);
    }
}
