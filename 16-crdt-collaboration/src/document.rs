//! Collaborative document implementation.

use crate::crdt::{AttributeValue, Element, Operation, PositionId, VectorClock};
use crate::{ClientId, DocumentId, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Collaborative document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Document ID.
    pub id: DocumentId,
    /// Elements in the document.
    elements: BTreeMap<PositionId, Element>,
    /// Tombstones (deleted elements).
    tombstones: HashSet<PositionId>,
    /// Vector clock.
    pub vector_clock: VectorClock,
    /// Root element.
    root: PositionId,
    /// Next sequence number per client.
    seq_counters: HashMap<ClientId, u32>,
}

impl Document {
    /// Create a new document.
    pub fn new(id: DocumentId) -> Self {
        let root = PositionId::root();
        let mut elements = BTreeMap::new();
        elements.insert(root.clone(), Element::new(root.clone(), '\0', None));

        Self {
            id,
            elements,
            tombstones: HashSet::new(),
            vector_clock: VectorClock::new(),
            root,
            seq_counters: HashMap::new(),
        }
    }

    /// Generate a new position ID.
    pub fn generate_position(&mut self, client_id: ClientId) -> PositionId {
        let lamport = self.vector_clock.increment(client_id);
        let seq = self.seq_counters.entry(client_id).or_insert(0);
        *seq += 1;
        PositionId::new(lamport, client_id, *seq)
    }

    /// Insert a character.
    pub fn insert(
        &mut self,
        client_id: ClientId,
        after: PositionId,
        value: char,
        attributes: HashMap<String, AttributeValue>,
    ) -> Result<Operation> {
        let id = self.generate_position(client_id);

        let element = Element {
            id: id.clone(),
            value,
            left: Some(after.clone()),
            right: None,
            attributes: attributes
                .into_iter()
                .map(|(k, v)| (k, (id.lamport, v)))
                .collect(),
            deleted: false,
        };

        // Find insertion point and update neighbors
        self.insert_element(element)?;

        Ok(Operation::Insert {
            id,
            after,
            value,
            attributes: HashMap::new(),
        })
    }

    /// Insert element maintaining order.
    fn insert_element(&mut self, element: Element) -> Result<()> {
        let id = element.id.clone();
        let left_id = element.left.clone();

        // Update left neighbor's right pointer
        if let Some(ref left) = left_id {
            if let Some(_left_elem) = self.elements.get_mut(left) {
                // Find correct position among right siblings
                let mut insert_after = left.clone();
                while let Some(ref right) = self.elements.get(&insert_after).and_then(|e| e.right.clone()) {
                    if right > &id {
                        break;
                    }
                    insert_after = right.clone();
                }

                let old_right = self.elements.get(&insert_after)
                    .and_then(|e| e.right.clone());

                // Update pointers
                if let Some(elem) = self.elements.get_mut(&insert_after) {
                    elem.right = Some(id.clone());
                }

                let mut new_element = element;
                new_element.left = Some(insert_after);
                new_element.right = old_right.clone();

                if let Some(ref right_id) = old_right {
                    if let Some(right_elem) = self.elements.get_mut(right_id) {
                        right_elem.left = Some(id.clone());
                    }
                }

                self.elements.insert(id, new_element);
            }
        } else {
            self.elements.insert(id, element);
        }

        Ok(())
    }

    /// Delete a character.
    pub fn delete(&mut self, client_id: ClientId, id: PositionId) -> Result<Operation> {
        if !self.elements.contains_key(&id) {
            return Err(crate::Error::InvalidOperation("Element not found".into()));
        }

        let deleted_by = self.generate_position(client_id);

        if let Some(element) = self.elements.get_mut(&id) {
            element.deleted = true;
        }
        self.tombstones.insert(id.clone());

        Ok(Operation::Delete { id, deleted_by })
    }

    /// Apply an operation.
    pub fn apply(&mut self, operation: &Operation) -> Result<()> {
        match operation {
            Operation::Insert {
                id,
                after,
                value,
                attributes,
            } => {
                if self.elements.contains_key(id) {
                    // Idempotent - already applied
                    return Ok(());
                }

                let element = Element {
                    id: id.clone(),
                    value: *value,
                    left: Some(after.clone()),
                    right: None,
                    attributes: attributes
                        .iter()
                        .map(|(k, v)| (k.clone(), (id.lamport, v.clone())))
                        .collect(),
                    deleted: false,
                };

                self.insert_element(element)?;

                // Update vector clock
                let current = self.vector_clock.get(&id.client_id);
                if id.lamport > current {
                    self.vector_clock.merge(&{
                        let mut vc = VectorClock::new();
                        vc.set(id.client_id, id.lamport);
                        vc
                    });
                }
            }

            Operation::Delete { id, deleted_by } => {
                if let Some(element) = self.elements.get_mut(id) {
                    element.deleted = true;
                }
                self.tombstones.insert(id.clone());

                // Update vector clock
                let current = self.vector_clock.get(&deleted_by.client_id);
                if deleted_by.lamport > current {
                    self.vector_clock.merge(&{
                        let mut vc = VectorClock::new();
                        vc.set(deleted_by.client_id, deleted_by.lamport);
                        vc
                    });
                }
            }

            Operation::Format {
                start,
                end,
                attribute,
                value,
            } => {
                let timestamp = start.lamport.max(end.lamport);

                // Find elements in range and update attribute
                for element in self.elements.values_mut() {
                    if &element.id >= start && &element.id <= end {
                        let entry = element.attributes.entry(attribute.clone())
                            .or_insert((0, AttributeValue::Null));
                        if timestamp > entry.0 {
                            *entry = (timestamp, value.clone());
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the document text.
    pub fn text(&self) -> String {
        let mut result = String::new();
        let mut current = self.root.clone();

        while let Some(element) = self.elements.get(&current) {
            if !element.deleted && element.value != '\0' {
                result.push(element.value);
            }
            match &element.right {
                Some(right) => current = right.clone(),
                None => break,
            }
        }

        result
    }

    /// Get the character at a given index.
    pub fn char_at(&self, index: usize) -> Option<char> {
        let mut current = self.root.clone();
        let mut count = 0;

        while let Some(element) = self.elements.get(&current) {
            if !element.deleted && element.value != '\0' {
                if count == index {
                    return Some(element.value);
                }
                count += 1;
            }
            match &element.right {
                Some(right) => current = right.clone(),
                None => break,
            }
        }

        None
    }

    /// Get the position ID at a given index.
    pub fn position_at(&self, index: usize) -> Option<PositionId> {
        let mut current = self.root.clone();
        let mut count = 0;

        // For index 0, return root
        if index == 0 {
            return Some(self.root.clone());
        }

        while let Some(element) = self.elements.get(&current) {
            if !element.deleted && element.value != '\0' {
                count += 1;
                if count == index {
                    return Some(element.id.clone());
                }
            }
            match &element.right {
                Some(right) => current = right.clone(),
                None => break,
            }
        }

        // Return last position if index is at end
        if count == index - 1 {
            return Some(current);
        }

        None
    }

    /// Get document length.
    pub fn len(&self) -> usize {
        self.elements
            .values()
            .filter(|e| !e.deleted && e.value != '\0')
            .count()
    }

    /// Check if document is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Garbage collect old tombstones.
    pub fn gc_tombstones(&mut self, min_vector_clock: &VectorClock) {
        self.tombstones.retain(|id| {
            !min_vector_clock.dominates(&{
                let mut vc = VectorClock::new();
                vc.set(id.client_id, id.lamport);
                vc
            })
        });
    }

    /// Get state snapshot for serialization.
    pub fn snapshot(&self) -> DocumentSnapshot {
        DocumentSnapshot {
            id: self.id,
            elements: self.elements.clone(),
            vector_clock: self.vector_clock.clone(),
            content: self.text(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Restore from snapshot.
    pub fn from_snapshot(snapshot: DocumentSnapshot) -> Self {
        let root = PositionId::root();
        Self {
            id: snapshot.id,
            elements: snapshot.elements,
            tombstones: HashSet::new(),
            vector_clock: snapshot.vector_clock,
            root,
            seq_counters: HashMap::new(),
        }
    }
}

/// Document snapshot for storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSnapshot {
    /// Document ID.
    pub id: DocumentId,
    /// Elements.
    pub elements: BTreeMap<PositionId, Element>,
    /// Vector clock.
    pub vector_clock: VectorClock,
    /// Serialized content for quick access.
    pub content: String,
    /// Timestamp when snapshot was taken.
    pub timestamp: u64,
}

/// Document metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Document ID.
    pub id: DocumentId,
    /// Title.
    pub title: String,
    /// Owner.
    pub owner: ClientId,
    /// Created timestamp.
    pub created_at: u64,
    /// Updated timestamp.
    pub updated_at: u64,
    /// Current version.
    pub version: u64,
    /// Whether the document is archived.
    pub archived: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insert() {
        let mut doc = Document::new(DocumentId::new_v4());
        let client = ClientId::new_v4();

        doc.insert(client, doc.root.clone(), 'H', HashMap::new()).unwrap();
        doc.insert(client, doc.position_at(1).unwrap(), 'i', HashMap::new()).unwrap();

        assert_eq!(doc.text(), "Hi");
    }

    #[test]
    fn test_concurrent_inserts() {
        let mut doc1 = Document::new(DocumentId::new_v4());
        let mut doc2 = doc1.clone();

        let client1 = ClientId::new_v4();
        let client2 = ClientId::new_v4();

        let op1 = doc1.insert(client1, doc1.root.clone(), 'A', HashMap::new()).unwrap();
        let op2 = doc2.insert(client2, doc2.root.clone(), 'B', HashMap::new()).unwrap();

        // Apply in different orders
        doc1.apply(&op2).unwrap();
        doc2.apply(&op1).unwrap();

        // Should converge
        assert_eq!(doc1.text(), doc2.text());
    }

    #[test]
    fn test_delete() {
        let mut doc = Document::new(DocumentId::new_v4());
        let client = ClientId::new_v4();

        let op = doc.insert(client, doc.root.clone(), 'X', HashMap::new()).unwrap();
        if let Operation::Insert { id, .. } = op {
            doc.delete(client, id).unwrap();
        }

        assert_eq!(doc.text(), "");
    }
}
