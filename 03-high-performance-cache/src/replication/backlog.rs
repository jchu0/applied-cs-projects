//! Replication backlog for partial resynchronization
//!
//! Circular buffer that stores recent commands for partial resync.

/// Replication backlog using a circular buffer
pub struct ReplicationBacklog {
    /// Buffer data
    buffer: Vec<u8>,
    /// Maximum size
    capacity: usize,
    /// Current write position
    write_pos: usize,
    /// Start offset in replication stream
    start_offset: u64,
    /// Total bytes written (used to calculate current offset)
    total_written: u64,
    /// Whether buffer has wrapped
    wrapped: bool,
}

impl ReplicationBacklog {
    /// Create a new backlog with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0; capacity],
            capacity,
            write_pos: 0,
            start_offset: 0,
            total_written: 0,
            wrapped: false,
        }
    }

    /// Append data to the backlog
    pub fn append(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }

        let data_len = data.len();

        // Update start offset if we'll overwrite old data
        if self.total_written + data_len as u64 > self.capacity as u64 {
            if !self.wrapped {
                self.wrapped = true;
            }
            // Calculate how much old data will be overwritten
            let total_after = self.total_written + data_len as u64;
            self.start_offset = total_after.saturating_sub(self.capacity as u64);
        }

        // Write data to circular buffer
        let mut remaining = data;
        while !remaining.is_empty() {
            let available = self.capacity - self.write_pos;
            let to_write = remaining.len().min(available);

            self.buffer[self.write_pos..self.write_pos + to_write]
                .copy_from_slice(&remaining[..to_write]);

            self.write_pos = (self.write_pos + to_write) % self.capacity;
            remaining = &remaining[to_write..];
        }

        self.total_written += data_len as u64;
    }

    /// Check if the given offset is available in the backlog
    pub fn contains_offset(&self, offset: u64) -> bool {
        if self.total_written == 0 {
            return false;
        }

        let end_offset = self.total_written;
        offset >= self.start_offset && offset <= end_offset
    }

    /// Get data from the given offset to current position
    pub fn get_from_offset(&self, offset: u64) -> Option<Vec<u8>> {
        if !self.contains_offset(offset) {
            return None;
        }

        let end_offset = self.total_written;
        let data_len = (end_offset - offset) as usize;

        if data_len == 0 {
            return Some(Vec::new());
        }

        if data_len > self.capacity {
            return None;
        }

        let mut result = Vec::with_capacity(data_len);

        // Calculate start position in buffer
        let bytes_in_buffer = if self.wrapped {
            self.capacity
        } else {
            self.total_written as usize
        };

        let offset_in_buffer = if self.wrapped {
            let from_start = (offset - self.start_offset) as usize;
            (self.write_pos + self.capacity - bytes_in_buffer + from_start) % self.capacity
        } else {
            (offset - self.start_offset) as usize
        };

        // Read data from circular buffer
        let mut pos = offset_in_buffer;
        let mut remaining = data_len;

        while remaining > 0 {
            let available = self.capacity - pos;
            let to_read = remaining.min(available);

            result.extend_from_slice(&self.buffer[pos..pos + to_read]);

            pos = (pos + to_read) % self.capacity;
            remaining -= to_read;
        }

        Some(result)
    }

    /// Current offset (end of backlog)
    pub fn current_offset(&self) -> u64 {
        self.total_written
    }

    /// Available data size
    pub fn len(&self) -> usize {
        if self.wrapped {
            self.capacity
        } else {
            self.total_written as usize
        }
    }

    /// Check if backlog is empty
    pub fn is_empty(&self) -> bool {
        self.total_written == 0
    }

    /// Clear the backlog
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.start_offset = 0;
        self.total_written = 0;
        self.wrapped = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_and_get() {
        let mut backlog = ReplicationBacklog::new(100);

        let data = b"SET key value\r\n";
        backlog.append(data);

        assert!(backlog.contains_offset(0));
        assert!(backlog.contains_offset(data.len() as u64));

        let retrieved = backlog.get_from_offset(0).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_partial_get() {
        let mut backlog = ReplicationBacklog::new(100);

        backlog.append(b"first\r\n");
        backlog.append(b"second\r\n");

        let offset = 7; // after "first\r\n"
        let data = backlog.get_from_offset(offset).unwrap();
        assert_eq!(data, b"second\r\n");
    }

    #[test]
    fn test_circular_wrap() {
        let mut backlog = ReplicationBacklog::new(10);

        // Write more than capacity
        backlog.append(b"12345");
        backlog.append(b"67890");
        backlog.append(b"abcde");

        // Old data should be gone
        assert!(!backlog.contains_offset(0));
        assert!(!backlog.contains_offset(4));

        // New data should be available
        assert!(backlog.contains_offset(10));
        assert!(backlog.contains_offset(15));
    }

    #[test]
    fn test_empty_backlog() {
        let backlog = ReplicationBacklog::new(100);
        assert!(backlog.is_empty());
        assert!(!backlog.contains_offset(0));
    }
}
