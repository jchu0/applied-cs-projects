//! Integration tests for CRDT collaboration system.

use crdt_collaboration::crdt::*;
use crdt_collaboration::document::Document;
use crdt_collaboration::{ClientId, DocumentId};
use std::collections::HashMap;

/// Helper to create deterministic client IDs for testing.
fn client_id(n: u8) -> ClientId {
    ClientId::from_bytes([n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
}

/// Helper to create deterministic document IDs for testing.
fn doc_id(n: u8) -> DocumentId {
    DocumentId::from_bytes([n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
}

#[cfg(test)]
mod document_tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let doc = Document::new(doc_id(1));
        assert_eq!(doc.id, doc_id(1));
        assert_eq!(doc.text(), "");
    }

    #[test]
    fn test_document_insert_single_char() {
        let mut doc = Document::new(doc_id(1));
        let client = client_id(1);
        let root = PositionId::root();

        doc.insert(client, root, 'H', HashMap::new()).unwrap();
        assert_eq!(doc.text(), "H");
    }

    #[test]
    fn test_document_insert_string() {
        let mut doc = Document::new(doc_id(1));
        let client = client_id(1);

        // Insert "Hello" character by character
        let mut after = PositionId::root();
        for ch in "Hello".chars() {
            let op = doc.insert(client, after.clone(), ch, HashMap::new()).unwrap();
            if let crdt_collaboration::crdt::Operation::Insert { id, .. } = op {
                after = id;
            }
        }

        assert_eq!(doc.text(), "Hello");
    }

    #[test]
    fn test_document_delete() {
        let mut doc = Document::new(doc_id(1));
        let client = client_id(1);

        // Insert "AB"
        let root = PositionId::root();
        let op_a = doc.insert(client, root.clone(), 'A', HashMap::new()).unwrap();
        let pos_a = if let crdt_collaboration::crdt::Operation::Insert { id, .. } = op_a {
            id
        } else {
            panic!("Expected Insert operation");
        };

        doc.insert(client, pos_a.clone(), 'B', HashMap::new()).unwrap();
        assert_eq!(doc.text(), "AB");

        // Delete 'A'
        doc.delete(client, pos_a).unwrap();
        assert_eq!(doc.text(), "B");
    }

    #[test]
    fn test_document_concurrent_inserts() {
        // Simulate two clients inserting concurrently
        let mut doc1 = Document::new(doc_id(1));
        let mut doc2 = Document::new(doc_id(1));
        let client1 = client_id(1);
        let client2 = client_id(2);
        let root = PositionId::root();

        // Client 1 inserts 'A'
        let op1 = doc1.insert(client1, root.clone(), 'A', HashMap::new()).unwrap();

        // Client 2 inserts 'B' (concurrently, starting from same root)
        let op2 = doc2.insert(client2, root.clone(), 'B', HashMap::new()).unwrap();

        // Apply op2 to doc1
        doc1.apply(&op2).unwrap();

        // Apply op1 to doc2
        doc2.apply(&op1).unwrap();

        // Both documents should converge to same content
        assert_eq!(doc1.text(), doc2.text());
    }
}

#[cfg(test)]
mod gcounter_integration_tests {
    use super::*;

    #[test]
    fn test_gcounter_distributed_increment() {
        let client1 = client_id(1);
        let client2 = client_id(2);
        let client3 = client_id(3);

        // Simulate three replicas
        let mut replica1 = GCounter::new();
        let mut replica2 = GCounter::new();
        let mut replica3 = GCounter::new();

        // Each replica increments locally
        replica1.increment(client1, 5);
        replica2.increment(client2, 3);
        replica3.increment(client3, 7);

        // Merge all replicas
        replica1.merge(&replica2);
        replica1.merge(&replica3);

        replica2.merge(&replica1);
        replica2.merge(&replica3);

        replica3.merge(&replica1);
        replica3.merge(&replica2);

        // All replicas should converge to same value
        assert_eq!(replica1.value(), 15);
        assert_eq!(replica2.value(), 15);
        assert_eq!(replica3.value(), 15);
    }

    #[test]
    fn test_gcounter_merge_idempotent() {
        let client = client_id(1);

        let mut counter1 = GCounter::new();
        let mut counter2 = GCounter::new();

        counter1.increment(client, 5);
        counter2.increment(client, 3);

        // Merge multiple times
        counter1.merge(&counter2);
        let v1 = counter1.value();

        counter1.merge(&counter2);
        let v2 = counter1.value();

        counter1.merge(&counter2);
        let v3 = counter1.value();

        assert_eq!(v1, v2);
        assert_eq!(v2, v3);
    }

    #[test]
    fn test_gcounter_merge_commutative() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        let mut counter_a = GCounter::new();
        let mut counter_b = GCounter::new();

        counter_a.increment(client1, 5);
        counter_b.increment(client2, 3);

        // Merge order 1: a <- b
        let mut merged1 = counter_a.clone();
        merged1.merge(&counter_b);

        // Merge order 2: b <- a
        let mut merged2 = counter_b.clone();
        merged2.merge(&counter_a);

        assert_eq!(merged1.value(), merged2.value());
    }
}

#[cfg(test)]
mod pncounter_integration_tests {
    use super::*;

    #[test]
    fn test_pncounter_distributed_operations() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        let mut replica1 = PNCounter::new();
        let mut replica2 = PNCounter::new();

        // Replica 1: +10, -3
        replica1.increment(client1, 10);
        replica1.decrement(client1, 3);

        // Replica 2: +5, -2
        replica2.increment(client2, 5);
        replica2.decrement(client2, 2);

        // Merge
        replica1.merge(&replica2);
        replica2.merge(&replica1);

        // Both should be: (10 - 3) + (5 - 2) = 7 + 3 = 10
        assert_eq!(replica1.value(), 10);
        assert_eq!(replica2.value(), 10);
    }

    #[test]
    fn test_pncounter_concurrent_decrement() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        let mut replica1 = PNCounter::new();
        let mut replica2 = PNCounter::new();

        // Both start from 0, both decrement
        replica1.decrement(client1, 5);
        replica2.decrement(client2, 3);

        replica1.merge(&replica2);
        replica2.merge(&replica1);

        assert_eq!(replica1.value(), -8);
        assert_eq!(replica2.value(), -8);
    }
}

#[cfg(test)]
mod lwwregister_integration_tests {
    use super::*;

    #[test]
    fn test_lww_register_convergence() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        // Create registers with different initial values and timestamps
        let mut reg1 = LWWRegister::new("old".to_string(), 1, client1);
        let reg2 = LWWRegister::new("new".to_string(), 2, client2);

        // Merge: reg2 has higher timestamp, should win
        reg1.merge(&reg2);
        assert_eq!(reg1.get(), "new");
    }

    #[test]
    fn test_lww_register_conflict_resolution() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        // Same timestamp, different clients - deterministic resolution
        let reg1 = LWWRegister::new("value1".to_string(), 5, client1);
        let reg2 = LWWRegister::new("value2".to_string(), 5, client2);

        // Merge in both directions
        let mut merged1 = reg1.clone();
        merged1.merge(&reg2);

        let mut merged2 = reg2.clone();
        merged2.merge(&reg1);

        // Should converge to same value (higher client_id wins)
        assert_eq!(merged1.get(), merged2.get());
    }
}

#[cfg(test)]
mod vector_clock_integration_tests {
    use super::*;

    #[test]
    fn test_vector_clock_causality_chain() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        let mut vc_a = VectorClock::new();
        let mut vc_b = VectorClock::new();
        let mut vc_c = VectorClock::new();

        // Event A at client1
        vc_a.increment(client1);

        // B receives A's clock and does work
        vc_b.merge(&vc_a);
        vc_b.increment(client2);

        // C receives B's clock
        vc_c.merge(&vc_b);
        vc_c.increment(client1);

        // Causality: A -> B -> C
        assert!(vc_a.happens_before(&vc_b));
        assert!(vc_b.happens_before(&vc_c));
        assert!(vc_a.happens_before(&vc_c)); // Transitivity
    }

    #[test]
    fn test_vector_clock_concurrent_detection() {
        let client1 = client_id(1);
        let client2 = client_id(2);

        let mut vc1 = VectorClock::new();
        let mut vc2 = VectorClock::new();

        // Independent operations
        vc1.increment(client1);
        vc2.increment(client2);

        // Should be concurrent
        assert!(vc1.is_concurrent(&vc2));
        assert!(!vc1.happens_before(&vc2));
        assert!(!vc2.happens_before(&vc1));
    }
}

#[cfg(test)]
mod position_id_tests {
    use super::*;

    #[test]
    fn test_position_id_total_ordering() {
        let client1 = client_id(1);
        let client2 = client_id(2);
        let client3 = client_id(3);

        let mut positions = vec![
            PositionId::new(1, client2, 0),
            PositionId::new(2, client1, 0),
            PositionId::new(1, client1, 0),
            PositionId::new(1, client3, 1),
            PositionId::new(1, client1, 1),
        ];

        // Should be sortable
        positions.sort();

        // Verify sorting is consistent
        for i in 0..positions.len() - 1 {
            assert!(positions[i] < positions[i + 1]);
        }
    }

    #[test]
    fn test_position_id_deterministic() {
        let client = client_id(1);

        let pos1 = PositionId::new(5, client, 3);
        let pos2 = PositionId::new(5, client, 3);

        assert_eq!(pos1, pos2);
    }
}
