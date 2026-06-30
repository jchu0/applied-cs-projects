//! Core CRDT implementations.
//!
//! Includes Vector Clocks, Position IDs, and sequence CRDTs.

use crate::{ClientId, Timestamp};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

/// Vector clock for tracking causality.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct VectorClock {
    /// Clock values per client.
    clocks: HashMap<ClientId, Timestamp>,
}

impl VectorClock {
    /// Create a new empty vector clock.
    pub fn new() -> Self {
        Self {
            clocks: HashMap::new(),
        }
    }

    /// Increment the clock for a client.
    pub fn increment(&mut self, client_id: ClientId) -> Timestamp {
        let entry = self.clocks.entry(client_id).or_insert(0);
        *entry += 1;
        *entry
    }

    /// Get the timestamp for a client.
    pub fn get(&self, client_id: &ClientId) -> Timestamp {
        self.clocks.get(client_id).copied().unwrap_or(0)
    }

    /// Set the timestamp for a client directly.
    pub fn set(&mut self, client_id: ClientId, timestamp: Timestamp) {
        self.clocks.insert(client_id, timestamp);
    }

    /// Merge with another vector clock.
    pub fn merge(&mut self, other: &VectorClock) {
        for (client, &time) in &other.clocks {
            let entry = self.clocks.entry(*client).or_insert(0);
            *entry = (*entry).max(time);
        }
    }

    /// Check if this clock happens before another.
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let mut dominated = false;

        for (client, &time) in &self.clocks {
            let other_time = other.clocks.get(client).copied().unwrap_or(0);
            if time > other_time {
                return false;
            }
            if time < other_time {
                dominated = true;
            }
        }

        // Also check for keys in other that we don't have
        for (client, &time) in &other.clocks {
            if !self.clocks.contains_key(client) && time > 0 {
                dominated = true;
            }
        }

        dominated
    }

    /// Check if this clock is concurrent with another.
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }

    /// Get all client IDs in this clock.
    pub fn clients(&self) -> Vec<ClientId> {
        self.clocks.keys().copied().collect()
    }

    /// Check if this clock dominates another (>=).
    pub fn dominates(&self, other: &VectorClock) -> bool {
        for (client, &time) in &other.clocks {
            if self.get(client) < time {
                return false;
            }
        }
        true
    }
}

/// Position identifier for CRDT elements.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PositionId {
    /// Lamport timestamp.
    pub lamport: Timestamp,
    /// Client that created this position.
    pub client_id: ClientId,
    /// Sequence number within same lamport.
    pub seq: u32,
}

impl PositionId {
    /// Create a new position ID.
    pub fn new(lamport: Timestamp, client_id: ClientId, seq: u32) -> Self {
        Self {
            lamport,
            client_id,
            seq,
        }
    }

    /// Root position (beginning of document).
    pub fn root() -> Self {
        Self {
            lamport: 0,
            client_id: ClientId::nil(),
            seq: 0,
        }
    }
}

impl Ord for PositionId {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.lamport.cmp(&other.lamport) {
            Ordering::Equal => match self.client_id.cmp(&other.client_id) {
                Ordering::Equal => self.seq.cmp(&other.seq),
                ord => ord,
            },
            ord => ord,
        }
    }
}

impl PartialOrd for PositionId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Attribute value for rich text.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AttributeValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
}

/// CRDT operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    /// Insert a character.
    Insert {
        id: PositionId,
        after: PositionId,
        value: char,
        attributes: HashMap<String, AttributeValue>,
    },
    /// Delete a character.
    Delete {
        id: PositionId,
        deleted_by: PositionId,
    },
    /// Format a range.
    Format {
        start: PositionId,
        end: PositionId,
        attribute: String,
        value: AttributeValue,
    },
}

impl Operation {
    /// Get the position ID of this operation.
    pub fn position_id(&self) -> &PositionId {
        match self {
            Operation::Insert { id, .. } => id,
            Operation::Delete { id, .. } => id,
            Operation::Format { start, .. } => start,
        }
    }
}

/// Element in the CRDT sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Element {
    /// Position ID.
    pub id: PositionId,
    /// Character value.
    pub value: char,
    /// Left neighbor.
    pub left: Option<PositionId>,
    /// Right neighbor.
    pub right: Option<PositionId>,
    /// Attributes (timestamp, value).
    pub attributes: HashMap<String, (Timestamp, AttributeValue)>,
    /// Whether this element is deleted.
    pub deleted: bool,
}

impl Element {
    /// Create a new element.
    pub fn new(id: PositionId, value: char, left: Option<PositionId>) -> Self {
        Self {
            id,
            value,
            left,
            right: None,
            attributes: HashMap::new(),
            deleted: false,
        }
    }
}

/// G-Counter (grow-only counter).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GCounter {
    counts: HashMap<ClientId, u64>,
}

impl GCounter {
    /// Create a new counter.
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }

    /// Increment the counter for a client.
    pub fn increment(&mut self, client_id: ClientId, amount: u64) {
        *self.counts.entry(client_id).or_insert(0) += amount;
    }

    /// Get the total value.
    pub fn value(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Merge with another counter.
    pub fn merge(&mut self, other: &GCounter) {
        for (client, &count) in &other.counts {
            let entry = self.counts.entry(*client).or_insert(0);
            *entry = (*entry).max(count);
        }
    }
}

/// PN-Counter (positive-negative counter).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PNCounter {
    /// Positive counter.
    positive: GCounter,
    /// Negative counter.
    negative: GCounter,
}

impl PNCounter {
    /// Create a new counter.
    pub fn new() -> Self {
        Self {
            positive: GCounter::new(),
            negative: GCounter::new(),
        }
    }

    /// Increment the counter.
    pub fn increment(&mut self, client_id: ClientId, amount: u64) {
        self.positive.increment(client_id, amount);
    }

    /// Decrement the counter.
    pub fn decrement(&mut self, client_id: ClientId, amount: u64) {
        self.negative.increment(client_id, amount);
    }

    /// Get the value.
    pub fn value(&self) -> i64 {
        self.positive.value() as i64 - self.negative.value() as i64
    }

    /// Merge with another counter.
    pub fn merge(&mut self, other: &PNCounter) {
        self.positive.merge(&other.positive);
        self.negative.merge(&other.negative);
    }
}

/// Last-Writer-Wins Register.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LWWRegister<T> {
    value: T,
    timestamp: Timestamp,
    client_id: ClientId,
}

impl<T: Clone> LWWRegister<T> {
    /// Create a new register.
    pub fn new(value: T, timestamp: Timestamp, client_id: ClientId) -> Self {
        Self {
            value,
            timestamp,
            client_id,
        }
    }

    /// Set the value.
    pub fn set(&mut self, value: T, timestamp: Timestamp, client_id: ClientId) {
        if timestamp > self.timestamp
            || (timestamp == self.timestamp && client_id > self.client_id)
        {
            self.value = value;
            self.timestamp = timestamp;
            self.client_id = client_id;
        }
    }

    /// Get the value.
    pub fn get(&self) -> &T {
        &self.value
    }

    /// Merge with another register.
    pub fn merge(&mut self, other: &LWWRegister<T>) {
        self.set(other.value.clone(), other.timestamp, other.client_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_clock_happens_before() {
        let mut vc1 = VectorClock::new();
        let mut vc2 = VectorClock::new();

        let client1 = ClientId::new_v4();
        let client2 = ClientId::new_v4();

        vc1.increment(client1);
        vc2.increment(client1);
        vc2.increment(client2);

        assert!(vc1.happens_before(&vc2));
        assert!(!vc2.happens_before(&vc1));
    }

    #[test]
    fn test_vector_clock_concurrent() {
        let mut vc1 = VectorClock::new();
        let mut vc2 = VectorClock::new();

        let client1 = ClientId::new_v4();
        let client2 = ClientId::new_v4();

        vc1.increment(client1);
        vc2.increment(client2);

        assert!(vc1.is_concurrent(&vc2));
    }

    #[test]
    fn test_pn_counter() {
        let mut counter = PNCounter::new();
        let client = ClientId::new_v4();

        counter.increment(client, 5);
        counter.decrement(client, 2);

        assert_eq!(counter.value(), 3);
    }

    #[test]
    fn test_lww_register() {
        let client1 = ClientId::new_v4();
        let client2 = ClientId::new_v4();

        let mut reg = LWWRegister::new("initial".to_string(), 1, client1);
        reg.set("second".to_string(), 2, client2);
        reg.set("old".to_string(), 1, client1); // Should be ignored

        assert_eq!(reg.get(), "second");
    }
}
