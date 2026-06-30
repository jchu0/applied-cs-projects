use std::time::Instant;

/// Redis object types
#[derive(Debug, Clone)]
pub enum RedisObject {
    String(StringObject),
    List(Vec<Vec<u8>>),
    Set(std::collections::HashSet<Vec<u8>>),
    Hash(std::collections::HashMap<Vec<u8>, Vec<u8>>),
    ZSet(ZSetObject),
}

/// String object - can be raw bytes or encoded integer
#[derive(Debug, Clone)]
pub enum StringObject {
    Raw(Vec<u8>),
    Int(i64),
}

impl StringObject {
    /// Create from bytes
    pub fn from_bytes(data: Vec<u8>) -> Self {
        // Try to encode as integer if possible
        if let Ok(s) = std::str::from_utf8(&data) {
            if let Ok(i) = s.parse::<i64>() {
                return StringObject::Int(i);
            }
        }
        StringObject::Raw(data)
    }

    /// Get as bytes
    pub fn as_bytes(&self) -> Vec<u8> {
        match self {
            StringObject::Raw(data) => data.clone(),
            StringObject::Int(i) => i.to_string().into_bytes(),
        }
    }

    /// Get length
    pub fn len(&self) -> usize {
        match self {
            StringObject::Raw(data) => data.len(),
            StringObject::Int(i) => i.to_string().len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Try to get as integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            StringObject::Int(i) => Some(*i),
            StringObject::Raw(data) => {
                std::str::from_utf8(data).ok()?.parse().ok()
            }
        }
    }

    /// Increment by amount
    pub fn incr(&mut self, amount: i64) -> Result<i64, &'static str> {
        let current = self.as_int().ok_or("value is not an integer")?;
        let new_value = current.checked_add(amount)
            .ok_or("increment would overflow")?;
        *self = StringObject::Int(new_value);
        Ok(new_value)
    }

    /// Append bytes
    pub fn append(&mut self, data: &[u8]) -> usize {
        let mut bytes = self.as_bytes();
        bytes.extend_from_slice(data);
        let new_len = bytes.len();
        *self = StringObject::from_bytes(bytes);
        new_len
    }
}

/// Sorted set object with skip list and hash map
#[derive(Debug, Clone)]
pub struct ZSetObject {
    pub dict: std::collections::HashMap<Vec<u8>, f64>,
    // In a full implementation, this would be a skip list
    pub sorted: Vec<(f64, Vec<u8>)>,
}

impl ZSetObject {
    pub fn new() -> Self {
        Self {
            dict: std::collections::HashMap::new(),
            sorted: Vec::new(),
        }
    }

    pub fn add(&mut self, score: f64, member: Vec<u8>) -> bool {
        let is_new = !self.dict.contains_key(&member);

        if let Some(old_score) = self.dict.insert(member.clone(), score) {
            // Remove old entry from sorted list
            self.sorted.retain(|(s, m)| !(*s == old_score && *m == member));
        }

        // Insert into sorted list
        let pos = self.sorted.partition_point(|(s, _)| *s < score);
        self.sorted.insert(pos, (score, member));

        is_new
    }

    pub fn remove(&mut self, member: &[u8]) -> bool {
        if let Some(score) = self.dict.remove(member) {
            self.sorted.retain(|(s, m)| !(*s == score && m == member));
            true
        } else {
            false
        }
    }

    pub fn score(&self, member: &[u8]) -> Option<f64> {
        self.dict.get(member).copied()
    }

    pub fn len(&self) -> usize {
        self.dict.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dict.is_empty()
    }
}

impl Default for ZSetObject {
    fn default() -> Self {
        Self::new()
    }
}

/// Entry metadata for LRU/LFU tracking
#[derive(Debug, Clone)]
pub struct EntryMetadata {
    /// Last access time for LRU
    pub lru_time: Instant,
    /// Access frequency for LFU (logarithmic counter)
    pub lfu_counter: u8,
    /// Time of last LFU decrement
    pub lfu_last_decr: u64,
}

impl Default for EntryMetadata {
    fn default() -> Self {
        Self {
            lru_time: Instant::now(),
            lfu_counter: 5, // LFU_INIT_VAL
            lfu_last_decr: 0,
        }
    }
}

impl EntryMetadata {
    /// Update LRU time on access
    pub fn touch(&mut self) {
        self.lru_time = Instant::now();
    }

    /// Increment LFU counter with probabilistic increment
    pub fn increment_lfu(&mut self) {
        if self.lfu_counter == 255 {
            return;
        }

        let r: f64 = rand::random();
        let base_val = (self.lfu_counter.saturating_sub(5)) as f64;
        let p = 1.0 / (base_val * 10.0 + 1.0);

        if r < p {
            self.lfu_counter = self.lfu_counter.saturating_add(1);
        }
    }
}
